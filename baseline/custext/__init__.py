"""CusText and CusText+ sanitization baselines.

The implementation follows Chen et al. (Findings of ACL 2023).  See the
package README for the distinction between the paper-faithful word-vector
protocol and the model-token adaptation used for like-for-like DP-SPD runs.
"""

from .sanitizer import CusTextConfig, CusTextSanitizer, load_text_embeddings

__all__ = ["CusTextConfig", "CusTextSanitizer", "load_text_embeddings"]
