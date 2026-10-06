"""Resume shard-by-shard checkpoint download and bounded INT4 PLE conversion.

Preserves indexed non-PLE tensor bytes. Never keeps two full checkpoints.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import torch
from huggingface_hub import HfApi
from safetensors import safe_open

ROOT=Path(os.environ.get('FLASHNEXT_ROOT', '/opt/flashnext'))
OUT=ROOT/'model-awq-int4ple'
TMP=ROOT/'download-staging'
OUT.mkdir(exist_ok=True)
TMP.mkdir(exist_ok=True)
source=json.loads((ROOT/'checkpoint-source.json').read_text())
repo,revision=source['repo'],source['revision']
base=f'https://huggingface.co/{repo}/resolve/{revision}/'
torch.set_num_threads(4)

def get_json(name):
    r=requests.get(base+name,timeout=60); r.raise_for_status(); return r.json()

index=get_json('model.safetensors.index.json')
info=HfApi().model_info(repo,revision=revision,files_metadata=True)
files={s.rfilename:s for s in info.siblings}

def download(name,path):
    expected=files[name].size
    part=path.with_suffix(path.suffix+'.part')
    if expected>256*1024**2:
        block_size=64*1024**2
        state_path=path.with_suffix(path.suffix+'.ranges.json')
        total=(expected+block_size-1)//block_size
        if state_path.exists():
            completed=set(json.loads(state_path.read_text()))
        else:
            contiguous=part.stat().st_size if part.exists() else 0
            completed=set(range(contiguous//block_size))
            if contiguous==expected: completed=set(range(total))
            state_path.write_text(json.dumps(sorted(completed)))
        descriptor=os.open(part,os.O_CREAT|os.O_RDWR,0o600)
        def transfer(number):
            start=number*block_size
            end=min(start+block_size,expected)-1
            for retry in range(5):
                try:
                    with requests.get(base+name+f'?range={start}-{end}',
                                      headers={'Range':f'bytes={start}-{end}'},
                                      stream=True,timeout=(30,90)) as r:
                        r.raise_for_status()
                        if r.status_code!=206: raise RuntimeError('Range request ignored')
                        offset=start
                        for data in r.iter_content(1024*1024):
                            if offset+len(data)>end+1: raise RuntimeError('Oversized range')
                            view=memoryview(data)
                            while view:
                                written=os.pwrite(descriptor,view,offset)
                                offset+=written; view=view[written:]
                        if offset!=end+1: raise RuntimeError('Incomplete range')
                    return number
                except Exception:
                    if retry==4: raise
                    time.sleep(2*(retry+1))
        try:
            with ThreadPoolExecutor(max_workers=6) as pool:
                futures=[pool.submit(transfer,i) for i in range(total) if i not in completed]
                for future in as_completed(futures):
                    completed.add(future.result())
                    temporary=state_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(sorted(completed)))
                    temporary.replace(state_path)
                    if len(completed)%16==0:
                        print('TRANSFER',name,len(completed),'/',total,flush=True)
            os.fsync(descriptor)
        finally: os.close(descriptor)
        assert part.stat().st_size==expected
        if files[name].lfs:
            with part.open('rb') as stream: digest=hashlib.file_digest(stream,'sha256').hexdigest()
            assert digest==files[name].lfs.sha256, 'Source SHA256 mismatch'
        part.replace(path)
        state_path.unlink()
        return
    for attempt in range(5):
        try:
            offset=part.stat().st_size if part.exists() else 0
            if offset!=expected:
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with requests.get(base+name,headers=headers,stream=True,timeout=(30,120)) as r:
                    r.raise_for_status()
                    if offset and r.status_code!=206:
                        raise RuntimeError('Server ignored resume range')
                    with part.open('ab' if offset else 'wb') as out:
                        for block in r.iter_content(4*1024*1024):
                            out.write(block)
            assert part.stat().st_size==expected
            if files[name].lfs:
                digest=hashlib.file_digest(part.open('rb'),'sha256').hexdigest()
                assert digest==files[name].lfs.sha256, 'Source SHA256 mismatch'
            part.replace(path)
            return
        except Exception:
            if attempt==4: raise
            time.sleep(3*(attempt+1))

for name in files:
    if '/' not in name and (name.endswith('.json') or name.endswith('.jinja') or name in ('LICENSE','README.md','tokenizer.model')) and name!='model.safetensors.index.json':
        path=OUT/name
        if not path.exists(): download(name,path)

grouped={}
for key,filename in index['weight_map'].items(): grouped.setdefault(filename,[]).append(key)
manifest=dict(source=source,format='tinybeast-ple-int4-group16-fp16-scale-v1',shards={})
manifest_path=OUT/'conversion-manifest.json'
if manifest_path.exists(): manifest=json.loads(manifest_path.read_text())
new_index={'metadata':{'total_size':0},'weight_map':{}}

for filename,names in sorted(grouped.items(),key=lambda item:('official' in item[0],item[0])):
    final=OUT/filename
    if filename in manifest['shards'] and final.exists():
        entry=manifest['shards'][filename]
        new_index['metadata']['total_size']+=entry['tensor_bytes']
        new_index['weight_map'].update({name:filename for name in names})
        continue
    src=TMP/filename
    if shutil.disk_usage(ROOT).free<files[filename].size*2+2*1024**3:
        raise RuntimeError(f'Insufficient staging disk headroom for {filename}')
    print('DOWNLOAD',filename,files[filename].size,flush=True)
    if not src.exists(): download(filename,src)
    with src.open('rb') as stream:
        header_size=struct.unpack('<Q',stream.read(8))[0]
        header=json.loads(stream.read(header_size))
    new_header={'__metadata__':{'format':'pt'}}
    cursor=0
    for name in names:
        tensor=dict(header[name])
        if '.ngram_embedding.shard_' in name and name.endswith('.weight'):
            assert tensor['dtype']=='F8_E4M3' and tensor['shape'][1]==160
            tensor['dtype']='U8'; tensor['shape']=[tensor['shape'][0],100]
            size=tensor['shape'][0]*100
        else: size=tensor['data_offsets'][1]-tensor['data_offsets'][0]
        tensor['data_offsets']=[cursor,cursor+size]
        new_header[name]=tensor
        cursor+=size
    encoded=json.dumps(new_header,separators=(',',':')).encode()
    encoded+=b' '*((-len(encoded))%8)
    part=final.with_suffix('.safetensors.part')
    print('CONVERT',filename,'output_bytes',cursor,flush=True)
    with part.open('wb') as output, src.open('rb') as original, safe_open(src,framework='pt',device='cpu') as tensors:
        output.write(struct.pack('<Q',len(encoded))); output.write(encoded)
        for name in names:
            if new_header[name]['dtype']=='U8' and '.ngram_embedding.shard_' in name:
                view=tensors.get_slice(name)
                for row in range(0,header[name]['shape'][0],32768):
                    x=view[row:row+32768].float().reshape(-1,10,16)
                    scales=(x.abs().amax(-1)/7).clamp_min(2**-24).half()
                    q=((x/scales.float().unsqueeze(-1)).round().clamp(-7,7)+8).to(torch.uint8).reshape(-1,160)
                    packed=q[:,0::2] | (q[:,1::2]<<4)
                    data=torch.cat((packed,scales.view(torch.uint8)),dim=1)
                    output.write(data.numpy().tobytes())
            else:
                start,end=header[name]['data_offsets']
                original.seek(8+header_size+start)
                remaining=end-start
                while remaining:
                    data=original.read(min(4*1024*1024,remaining))
                    if not data: raise EOFError(name)
                    output.write(data); remaining-=len(data)
        output.flush(); os.fsync(output.fileno())
    part.replace(final)
    with final.open('rb') as result:
        digest=hashlib.file_digest(result,'sha256').hexdigest()
    manifest['shards'][filename]={'sha256':digest,'tensor_bytes':cursor}
    manifest_path.write_text(json.dumps(manifest,indent=2))
    src.unlink()  # Only the temporary source shard produced by this script.
    new_index['metadata']['total_size']+=cursor
    new_index['weight_map'].update({name:filename for name in names})
    print('COMPLETE',filename,'free_gib',round(shutil.disk_usage(ROOT).free/1024**3,2),flush=True)

(OUT/'model.safetensors.index.json').write_text(json.dumps(new_index,indent=2))
print('READY',OUT,'tensor_gib',new_index['metadata']['total_size']/1024**3,flush=True)
