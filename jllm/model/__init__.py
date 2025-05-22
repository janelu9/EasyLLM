from .llama.pipeline_llama import LlamaForCausalLMPipe,autopartition_transformer
from .baichuan.pipeline_baichuan import BaichuanForCausalLMPipe
from .qwen.pipeline_qwen import QWenForCausalLMPipe,QWenForClassCausalLMPipe
from .qwen2.pipeline_qwen2 import Qwen2ForCausalLMPipe
from .lora import convert_linear_layer_to_lora,_z3_params_to_fetch,convert_lora_to_linear_layer,only_optimize_lora_parameters,make_model_gradient_checkpointing_compatible
from .llama.parallel_llama import LlamaForCausalLMPipe as LlamaForCausalLMParallel
from .qwen2.parallel_qwen2 import Qwen2ForCausalLMPipe as Qwen2ForCausalLMParallel
from .qwen2.pipeline_qwen2_moe import Qwen2MoeForCausalLMPipe,get_num_hidden_layers as qwen_get_num_hidden_layers
from .qwen2_vl.pipeline_qwen2_vl import Qwen2VLForCausalLMPipe
from .qwen2_vl.parallel_qwen2_vl import Qwen2VLForCausalLMParallel
from .qwen2_5_vl.pipeline_qwen2_5_vl import Qwen2_5_VLForCausalLMPipe
from .qwen2_5_vl.parallel_qwen2_5_vl import Qwen2_5_VLForCausalLMParallel
from .intern.pipeline_internvl2 import InterenVL2ForCausalLMPipe,autopartition_decoder
from .intern.pipeline_internlm2 import InternLM2ForCausalLMPipe
from .deepseek_v3.parallel_deepseek_v3 import DeepseekV3ForCausalLM,get_num_hidden_layers as deepseek_v3_get_num_hidden_layers

ModelPipe = {
    'LlamaForCausalLM':LlamaForCausalLMPipe,
    'QWenLMHeadModel':QWenForCausalLMPipe,
    'Qwen2ForCausalLM':Qwen2ForCausalLMPipe,
    'QWenForClassCausalLM':QWenForClassCausalLMPipe,
    'BaichuanForCausalLM':BaichuanForCausalLMPipe,
    'Qwen2MoeForCausalLM':Qwen2MoeForCausalLMPipe,
    'InternVLChatModel':InterenVL2ForCausalLMPipe,
    'InternLM2ForCausalLM':InternLM2ForCausalLMPipe,
    'Qwen2VLForConditionalGeneration':Qwen2VLForCausalLMPipe,
    'Qwen2_5_VLForConditionalGeneration':Qwen2_5_VLForCausalLMPipe,
    'Qwen3ForCausalLM':Qwen2ForCausalLMPipe,
}
    
ModelParallel = {
    'LlamaForCausalLM':LlamaForCausalLMParallel,
    'Qwen2ForCausalLM':Qwen2ForCausalLMParallel,
    'Qwen2VLForConditionalGeneration':Qwen2VLForCausalLMParallel,
    'Qwen2_5_VLForConditionalGeneration':Qwen2_5_VLForCausalLMParallel,
    'DeepseekV3ForCausalLM':DeepseekV3ForCausalLM,
    'Qwen3ForCausalLM':Qwen2ForCausalLMParallel,
}

get_virtual_num_hidden_layers ={
    "Qwen2MoeForCausalLM":qwen_get_num_hidden_layers,
    "DeepseekV3ForCausalLM":deepseek_v3_get_num_hidden_layers
}

