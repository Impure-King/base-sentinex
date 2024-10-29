from typing import Tuple

# Taken from Keras:
def conv_dimension_detector(in_feature, out_features,
                            out_features_dim,
                            kernel_size):
    # Add support for multi-features:
    if len(kernel_size) > out_features_dim:
        variables = kernel_size[:out_features_dim]
        variables += (out_features,)
        variables += kernel_size[out_features_dim:]
        variables += (in_feature)
        
        
# Need to recheck and properly comment. I need my own alternative for this.