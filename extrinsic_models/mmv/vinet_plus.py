from mmv.models.mm_embeddings import AudioTextVideoEmbedding
from mmv import config
import jax.numpy as jnp, os, numpy as np
from typing import Any, Dict
from mmv.utils import checkpoint
import haiku as hk, jax, functools, torch
# os.environ['XLA_FLAGS'] = "--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=10 inter_op_parallelism_threads=10"
jax.config.update('jax_platform_name', 'cpu')

def forward_fn(images: jnp.ndarray, is_training: bool, final_endpoint: str, model_config: Dict[str, Any]):
    module = AudioTextVideoEmbedding(**model_config, word_embedding_matrix = None)
    """
    Args:
        images: A 5-D float array of shape `[B, T, H, W, 3]`.
        is_training: Whether to use training mode.
        final_endpoint: Up to which endpoint to run / return.
    """
    return module(images=images, 
                  is_training=is_training, 
                  final_endpoint=final_endpoint,
                  audio_spectrogram=None,
                  word_ids=None)['vid_repr']

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_PATH = os.path.join(CURRENT_PATH, 'mmv_tsm_resnet_x2.pkl')
model_config = config.get_model_config(CHECKPOINT_PATH)
pretrained_weights = checkpoint.load_checkpoint(CHECKPOINT_PATH)
params = pretrained_weights['params']
state = pretrained_weights['state']

forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))
forward_apply = jax.jit(functools.partial(forward.apply, is_training=False, final_endpoint='Embeddings', model_config=model_config, params=params, state=state))

def calculate_mmv_embeddings(video_data):
    vid_representation_test, _ = forward_apply(images=video_data)
    result = [torch.from_numpy(np.asarray(v)).permute(3, 0, 1, 2).cuda() for v in vid_representation_test]
    # print([r.shape for r in result]) # Every element is of shape: (channels, time_width, height, width)
    del vid_representation_test
    return result