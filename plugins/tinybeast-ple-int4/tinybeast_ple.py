"""Experimental INT4 PLE method for the pinned vllm-backport runtime.

Install with install() after importing ngram_embedding. FP8 checkpoint values
are quantized in small GPU chunks. The original global FP8 scale is preserved.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lookup(packed, scales, ids, output, TOTAL: tl.constexpr,
            WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = pos < TOTAL
    row = tl.load(ids + pos // WIDTH, mask, 0)
    col = pos % WIDTH
    byte = tl.load(packed + row * (WIDTH // 2) + col // 2, mask, 0).to(tl.int32)
    value = ((byte >> ((col % 2) * 4)) & 15) - 8
    scale = tl.load(scales + row * (WIDTH // 16) + col // 16, mask, 0)
    tl.store(output + pos, value.to(tl.float32) * scale.to(tl.float32), mask)


def load_rows(packed, scales, source, *, checkpoint_start, tp_start, tp_end,
              chunk_rows=4096):
    """Load only this rank's overlap, using bounded temporary GPU allocations."""
    if source.ndim == 2 and source.dtype == torch.uint8 and source.shape[1] == 100:
        begin=max(checkpoint_start,tp_start)
        end=min(checkpoint_start+len(source),tp_end)
        for start in range(begin,end,chunk_rows):
            stop=min(start+chunk_rows,end)
            block=source[start-checkpoint_start:stop-checkpoint_start]
            target=slice(start-tp_start,stop-tp_start)
            packed[target].copy_(block[:,:80])
            scales[target].copy_(block[:,80:].contiguous().view(torch.float16))
        return max(0,end-begin)
    if source.ndim != 2 or source.shape[1] != packed.shape[1] * 2:
        raise ValueError('Unexpected PLE row shape')
    if source.dtype != torch.float8_e4m3fn:
        raise ValueError('This conversion requires stored FP8 E4M3 values')
    begin = max(checkpoint_start, tp_start)
    end = min(checkpoint_start + source.shape[0], tp_end)
    width = source.shape[1]
    if width % 16:
        raise ValueError('PLE width must be divisible by group size 16')
    for start in range(begin, end, chunk_rows):
        stop = min(start + chunk_rows, end)
        x = source[start-checkpoint_start:stop-checkpoint_start].to(
            device=packed.device, dtype=torch.float32)
        grouped = x.reshape(-1, width // 16, 16)
        # Clamp to a representable FP16 value for all-zero groups.
        scale = (grouped.abs().amax(-1) / 7).clamp_min(2**-24).half()
        q = (grouped / scale.float().unsqueeze(-1)).round().clamp(-7,7)
        u = (q.reshape(-1,width) + 8).to(torch.uint8)
        target = slice(start-tp_start, stop-tp_start)
        packed[target].copy_(u[:,0::2] | (u[:,1::2] << 4))
        scales[target].copy_(scale)
    return max(0, end-begin)


def lookup(packed, scales, ids):
    width = packed.shape[1] * 2
    ids = ids.contiguous()
    output = torch.empty((*ids.shape,width),device=packed.device,dtype=torch.float16)
    if output.numel():
        _lookup[(triton.cdiv(output.numel(),256),)](
            packed,scales,ids,output,output.numel(),width,256)
    return output


def install():
    from vllm.models.qwen4_exp.nvidia import ngram_embedding as ng
    from vllm.model_executor.parameter import ModelWeightParameter
    if getattr(ng, '_tinybeast_int4_installed', False):
        return

    class Int4Method(ng.Qwen4ExpPLEFp8EmbeddingMethod):
        def create_weights(self,layer,input_size_per_partition,output_partition_sizes,
                           input_size,output_size,params_dtype,**attrs):
            if not isinstance(layer, ng.Qwen4ExpPLEDeviceEmbedding):
                raise ValueError('TinyBeast INT4 requires GPU-resident PLE; disable CPU offload')
            width = input_size_per_partition
            rows = sum(output_partition_sizes)
            if width != 160:
                raise ValueError(f'Unexpected embedding width: {width}')
            data = layer.allocate_embedding_weight(rows,width//2,torch.uint8)
            data.fill_(0x88)
            layer.register_buffer('int4_scales',torch.zeros(
                rows,width//16,device=data.device,dtype=torch.float16))

            def loader(param,loaded_weight,checkpoint_start=None):
                if checkpoint_start is None:
                    raise ValueError('Expected checkpoint-sharded PLE weights')
                bounds=layer.shard_indices
                load_rows(param,layer.int4_scales,loaded_weight,
                          checkpoint_start=checkpoint_start,
                          tp_start=bounds.org_vocab_start_index,
                          tp_end=bounds.org_vocab_end_index)

            layer.register_parameter('weight',ModelWeightParameter(
                data=data,input_dim=1,output_dim=0,weight_loader=loader))
            global_scale=ng.create_fp8_scale_parameter(
                ng.PerTensorScaleParameter,output_partition_sizes,
                input_size_per_partition,None,attrs.get('weight_loader'),
                scale_dtype=torch.float32)
            layer.register_parameter('weight_scale',global_scale)

        def embedding(self,layer,input_):
            return lookup(layer.weight,layer.int4_scales,input_)

    original=ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config

    def select(quant_config,prefix,embedding_dtype=None):
        method=original(quant_config,prefix,embedding_dtype)
        if isinstance(method,ng.Qwen4ExpPLEFp8EmbeddingMethod):
            return Int4Method()
        raise ValueError('TinyBeast PLE patch expects an FP8 PLE checkpoint')

    ng.Qwen4ExpPLEEmbeddingMethod.from_quant_config=staticmethod(select)
    original_load=ng.Qwen4ExpNGramEmbedding.load_weights

    def load_weights(self,weights):
        loaded=set()
        def regular():
            for name,value in weights:
                if (name.startswith('ngram_embedding.shard_') and name.endswith('.weight')
                        and value.dtype==torch.uint8):
                    index=int(name[len('ngram_embedding.shard_'):-len('.weight')])
                    embedding=self.ngram_embedding
                    shard_size=(embedding.org_vocab_size+self.split_ngram_parts-1)//self.split_ngram_parts
                    start=index*shard_size
                    expected=min(shard_size,embedding.org_vocab_size-start)
                    if index<0 or index>=self.split_ngram_parts or tuple(value.shape)!=(expected,100):
                        raise ValueError(f'Invalid packed PLE tensor {name}: {value.shape}')
                    embedding.weight.weight_loader(embedding.weight,value,checkpoint_start=start)
                    loaded.add('ngram_embedding.weight')
                else:
                    yield name,value
        loaded.update(original_load(self,regular()))
        return loaded

    ng.Qwen4ExpNGramEmbedding.load_weights=load_weights
    ng._tinybeast_int4_installed=True
