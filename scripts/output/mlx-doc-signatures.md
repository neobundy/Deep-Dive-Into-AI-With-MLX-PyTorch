mlx._reprlib_fix.FixedRepr()
mlx.extension.CMakeBuild(dist, **kw)
mlx.extension.CMakeExtension(name: str, sourcedir: str = '') -> None
mlx.extension.Extension(name, sources, *args, **kw)
mlx.extension.Path(*args, **kwargs)
mlx.extension.build_ext(dist, **kw)
mlx.extension.find_namespace_packages(where: Union[str, os.PathLike] = '.', exclude: Iterable[str] = (), include: Iterable[str] = ('*',)) -> List[str]
mlx.extension.setup(**attrs)
mlx.nn.ALiBi()
mlx.nn.BatchNorm(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)
mlx.nn.Bilinear(input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.CELU(alpha=1.0)
mlx.nn.Conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True)
mlx.nn.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, bias: bool = True)
mlx.nn.Dropout(p: float = 0.5)
mlx.nn.Dropout2d(p: float = 0.5)
mlx.nn.Dropout3d(p: float = 0.5)
mlx.nn.ELU(alpha=1.0)
mlx.nn.Embedding(num_embeddings: int, dims: int)
mlx.nn.GELU(approx='none')
mlx.nn.GLU(axis: int = -1)
mlx.nn.GroupNorm(num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False)
mlx.nn.Hardswish()
mlx.nn.Identity(*args: Any, **kwargs: Any) -> None
mlx.nn.InstanceNorm(dims: int, eps: float = 1e-05, affine: bool = False)
mlx.nn.LayerNorm(dims: int, eps: float = 1e-05, affine: bool = True)
mlx.nn.LeakyReLU(negative_slope=0.01)
mlx.nn.Linear(input_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.LogSigmoid()
mlx.nn.LogSoftmax()
mlx.nn.Mish()
mlx.nn.Module()
mlx.nn.MultiHeadAttention(dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims: Optional[int] = None, bias: bool = False)
mlx.nn.PReLU(num_parameters=1, init=0.25)
mlx.nn.QuantizedLinear(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4)
mlx.nn.RMSNorm(dims: int, eps: float = 1e-05)
mlx.nn.ReLU()
mlx.nn.ReLU6()
mlx.nn.RoPE(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0)
mlx.nn.SELU()
mlx.nn.Sequential(*modules)
mlx.nn.SiLU()
mlx.nn.Sigmoid()
mlx.nn.SinusoidalPositionalEncoding(dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False)
mlx.nn.Softmax()
mlx.nn.Softplus()
mlx.nn.Softsign()
mlx.nn.Step(threshold: float = 0.0)
mlx.nn.Tanh()
mlx.nn.Transformer(dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None, norm_first: bool = False)
mlx.nn.TransformerEncoder(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.TransformerEncoderLayer(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.activations.Any(*args, **kwargs)
mlx.nn.layers.activations.CELU(alpha=1.0)
mlx.nn.layers.activations.ELU(alpha=1.0)
mlx.nn.layers.activations.GELU(approx='none')
mlx.nn.layers.activations.GLU(axis: int = -1)
mlx.nn.layers.activations.Hardswish()
mlx.nn.layers.activations.LeakyReLU(negative_slope=0.01)
mlx.nn.layers.activations.LogSigmoid()
mlx.nn.layers.activations.LogSoftmax()
mlx.nn.layers.activations.Mish()
mlx.nn.layers.activations.Module()
mlx.nn.layers.activations.PReLU(num_parameters=1, init=0.25)
mlx.nn.layers.activations.ReLU()
mlx.nn.layers.activations.ReLU6()
mlx.nn.layers.activations.SELU()
mlx.nn.layers.activations.SiLU()
mlx.nn.layers.activations.Sigmoid()
mlx.nn.layers.activations.Softmax()
mlx.nn.layers.activations.Softplus()
mlx.nn.layers.activations.Softsign()
mlx.nn.layers.activations.Step(threshold: float = 0.0)
mlx.nn.layers.activations.Tanh()
mlx.nn.layers.activations.celu(x, alpha=1.0)
mlx.nn.layers.activations.elu(x, alpha=1.0)
mlx.nn.layers.activations.gelu(x)
mlx.nn.layers.activations.gelu_approx(x)
mlx.nn.layers.activations.gelu_fast_approx(x)
mlx.nn.layers.activations.glu(x: mlx.core.array, axis: int = -1) -> mlx.core.array
mlx.nn.layers.activations.hardswish(x)
mlx.nn.layers.activations.leaky_relu(x, negative_slope=0.01)
mlx.nn.layers.activations.log_sigmoid(x)
mlx.nn.layers.activations.log_softmax(x, axis=-1)
mlx.nn.layers.activations.mish(x: mlx.core.array) -> mlx.core.array
mlx.nn.layers.activations.prelu(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array
mlx.nn.layers.activations.relu(x)
mlx.nn.layers.activations.relu6(x)
mlx.nn.layers.activations.selu(x)
mlx.nn.layers.activations.sigmoid(x)
mlx.nn.layers.activations.silu(x)
mlx.nn.layers.activations.softmax(x, axis=-1)
mlx.nn.layers.activations.softplus(x)
mlx.nn.layers.activations.softsign(x)
mlx.nn.layers.activations.step(x: mlx.core.array, threshold: float = 0.0)
mlx.nn.layers.activations.tanh(x)
mlx.nn.layers.base.Any(*args, **kwargs)
mlx.nn.layers.base.Module()
mlx.nn.layers.base.tree_flatten(tree, prefix='', is_leaf=None)
mlx.nn.layers.base.tree_unflatten(tree)
mlx.nn.celu(x, alpha=1.0)
mlx.nn.layers.containers.Module()
mlx.nn.layers.containers.Sequential(*modules)
mlx.nn.layers.convolution.Conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True)
mlx.nn.layers.convolution.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, bias: bool = True)
mlx.nn.layers.convolution.Module()
mlx.nn.layers.dropout.Dropout(p: float = 0.5)
mlx.nn.layers.dropout.Dropout2d(p: float = 0.5)
mlx.nn.layers.dropout.Dropout3d(p: float = 0.5)
mlx.nn.layers.dropout.Module()
mlx.nn.elu(x, alpha=1.0)
mlx.nn.layers.embedding.Embedding(num_embeddings: int, dims: int)
mlx.nn.layers.embedding.Module()
mlx.nn.gelu(x)
mlx.nn.gelu_approx(x)
mlx.nn.gelu_fast_approx(x)
mlx.nn.glu(x: mlx.core.array, axis: int = -1) -> mlx.core.array
mlx.nn.hardswish(x)
mlx.nn.layers.ALiBi()
mlx.nn.layers.BatchNorm(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)
mlx.nn.layers.Bilinear(input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.CELU(alpha=1.0)
mlx.nn.layers.Conv1d(in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = True)
mlx.nn.layers.Conv2d(in_channels: int, out_channels: int, kernel_size: Union[int, tuple], stride: Union[int, tuple] = 1, padding: Union[int, tuple] = 0, bias: bool = True)
mlx.nn.layers.Dropout(p: float = 0.5)
mlx.nn.layers.Dropout2d(p: float = 0.5)
mlx.nn.layers.Dropout3d(p: float = 0.5)
mlx.nn.layers.ELU(alpha=1.0)
mlx.nn.layers.Embedding(num_embeddings: int, dims: int)
mlx.nn.layers.GELU(approx='none')
mlx.nn.layers.GLU(axis: int = -1)
mlx.nn.layers.GroupNorm(num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False)
mlx.nn.layers.Hardswish()
mlx.nn.layers.Identity(*args: Any, **kwargs: Any) -> None
mlx.nn.layers.InstanceNorm(dims: int, eps: float = 1e-05, affine: bool = False)
mlx.nn.layers.LayerNorm(dims: int, eps: float = 1e-05, affine: bool = True)
mlx.nn.layers.LeakyReLU(negative_slope=0.01)
mlx.nn.layers.Linear(input_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.LogSigmoid()
mlx.nn.layers.LogSoftmax()
mlx.nn.layers.Mish()
mlx.nn.layers.Module()
mlx.nn.layers.MultiHeadAttention(dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims: Optional[int] = None, bias: bool = False)
mlx.nn.layers.PReLU(num_parameters=1, init=0.25)
mlx.nn.layers.QuantizedLinear(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4)
mlx.nn.layers.RMSNorm(dims: int, eps: float = 1e-05)
mlx.nn.layers.ReLU()
mlx.nn.layers.ReLU6()
mlx.nn.layers.RoPE(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0)
mlx.nn.layers.SELU()
mlx.nn.layers.Sequential(*modules)
mlx.nn.layers.SiLU()
mlx.nn.layers.Sigmoid()
mlx.nn.layers.SinusoidalPositionalEncoding(dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False)
mlx.nn.layers.Softmax()
mlx.nn.layers.Softplus()
mlx.nn.layers.Softsign()
mlx.nn.layers.Step(threshold: float = 0.0)
mlx.nn.layers.Tanh()
mlx.nn.layers.Transformer(dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None, norm_first: bool = False)
mlx.nn.layers.TransformerEncoder(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.TransformerEncoderLayer(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.celu(x, alpha=1.0)
mlx.nn.layers.elu(x, alpha=1.0)
mlx.nn.layers.gelu(x)
mlx.nn.layers.gelu_approx(x)
mlx.nn.layers.gelu_fast_approx(x)
mlx.nn.layers.glu(x: mlx.core.array, axis: int = -1) -> mlx.core.array
mlx.nn.layers.hardswish(x)
mlx.nn.layers.leaky_relu(x, negative_slope=0.01)
mlx.nn.layers.linear.Any(*args, **kwargs)
mlx.nn.layers.linear.Bilinear(input1_dims: int, input2_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.linear.Identity(*args: Any, **kwargs: Any) -> None
mlx.nn.layers.linear.Linear(input_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.linear.Module()
mlx.nn.layers.log_sigmoid(x)
mlx.nn.layers.log_softmax(x, axis=-1)
mlx.nn.layers.mish(x: mlx.core.array) -> mlx.core.array
mlx.nn.layers.normalization.BatchNorm(num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = True, track_running_stats: bool = True)
mlx.nn.layers.normalization.GroupNorm(num_groups: int, dims: int, eps: float = 1e-05, affine: bool = True, pytorch_compatible: bool = False)
mlx.nn.layers.normalization.InstanceNorm(dims: int, eps: float = 1e-05, affine: bool = False)
mlx.nn.layers.normalization.LayerNorm(dims: int, eps: float = 1e-05, affine: bool = True)
mlx.nn.layers.normalization.Module()
mlx.nn.layers.normalization.RMSNorm(dims: int, eps: float = 1e-05)
mlx.nn.layers.positional_encoding.ALiBi()
mlx.nn.layers.positional_encoding.Module()
mlx.nn.layers.positional_encoding.RoPE(dims: int, traditional: bool = False, base: float = 10000, scale: float = 1.0)
mlx.nn.layers.positional_encoding.SinusoidalPositionalEncoding(dims: int, min_freq: float = 0.0001, max_freq: float = 1, scale: Optional[float] = None, cos_first: bool = False, full_turns: bool = False)
mlx.nn.layers.prelu(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array
mlx.nn.layers.quantized.Linear(input_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.quantized.Module()
mlx.nn.layers.quantized.QuantizedLinear(input_dims: int, output_dims: int, bias: bool = True, group_size: int = 64, bits: int = 4)
mlx.nn.layers.quantized.tree_flatten(tree, prefix='', is_leaf=None)
mlx.nn.layers.quantized.tree_map(fn, tree, *rest, is_leaf=None)
mlx.nn.layers.relu(x)
mlx.nn.layers.relu6(x)
mlx.nn.layers.selu(x)
mlx.nn.layers.silu(x)
mlx.nn.layers.softmax(x, axis=-1)
mlx.nn.layers.softplus(x)
mlx.nn.layers.softsign(x)
mlx.nn.layers.step(x: mlx.core.array, threshold: float = 0.0)
mlx.nn.layers.tanh(x)
mlx.nn.layers.transformer.Any(*args, **kwargs)
mlx.nn.layers.transformer.Dropout(p: float = 0.5)
mlx.nn.layers.transformer.LayerNorm(dims: int, eps: float = 1e-05, affine: bool = True)
mlx.nn.layers.transformer.Linear(input_dims: int, output_dims: int, bias: bool = True) -> None
mlx.nn.layers.transformer.Module()
mlx.nn.layers.transformer.MultiHeadAttention(dims: int, num_heads: int, query_input_dims: Optional[int] = None, key_input_dims: Optional[int] = None, value_input_dims: Optional[int] = None, value_dims: Optional[int] = None, value_output_dims: Optional[int] = None, bias: bool = False)
mlx.nn.layers.transformer.Transformer(dims: int = 512, num_heads: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, custom_encoder: Optional[Any] = None, custom_decoder: Optional[Any] = None, norm_first: bool = False)
mlx.nn.layers.transformer.TransformerDecoder(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.transformer.TransformerDecoderLayer(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.transformer.TransformerEncoder(num_layers: int, dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation=<function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.transformer.TransformerEncoderLayer(dims: int, num_heads: int, mlp_dims: Optional[int] = None, dropout: float = 0.0, activation: Callable[[Any], Any] = <function relu at 0x289c709a0>, norm_first: bool = False)
mlx.nn.layers.transformer.relu(x)
mlx.nn.leaky_relu(x, negative_slope=0.01)
mlx.nn.log_sigmoid(x)
mlx.nn.log_softmax(x, axis=-1)
mlx.nn.losses.binary_cross_entropy(logits: mlx.core.array, targets: mlx.core.array, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.cosine_similarity_loss(x1: mlx.core.array, x2: mlx.core.array, axis: int = 1, eps: float = 1e-08, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.cross_entropy(logits: mlx.core.array, targets: mlx.core.array, weights: mlx.core.array = None, axis: int = -1, label_smoothing: float = 0.0, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.gaussian_nll_loss(inputs: mlx.core.array, targets: mlx.core.array, vars: mlx.core.array, full: bool = False, eps: float = 1e-06, reduction: Literal['none', 'mean', 'sum'] = 'mean') -> mlx.core.array
mlx.nn.losses.hinge_loss(inputs: mlx.core.array, targets: mlx.core.array, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.huber_loss(inputs: mlx.core.array, targets: mlx.core.array, delta: float = 1.0, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.kl_div_loss(inputs: mlx.core.array, targets: mlx.core.array, axis: int = -1, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.l1_loss(predictions: mlx.core.array, targets: mlx.core.array, reduction: Literal['none', 'mean', 'sum'] = 'mean') -> mlx.core.array
mlx.nn.losses.log_cosh_loss(inputs: mlx.core.array, targets: mlx.core.array, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.mse_loss(predictions: mlx.core.array, targets: mlx.core.array, reduction: Literal['none', 'mean', 'sum'] = 'mean') -> mlx.core.array
mlx.nn.losses.nll_loss(inputs: mlx.core.array, targets: mlx.core.array, axis: int = -1, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.losses.smooth_l1_loss(predictions: mlx.core.array, targets: mlx.core.array, beta: float = 1.0, reduction: Literal['none', 'mean', 'sum'] = 'mean') -> mlx.core.array
mlx.nn.losses.triplet_loss(anchors: mlx.core.array, positives: mlx.core.array, negatives: mlx.core.array, axis: int = -1, p: int = 2, margin: float = 1.0, eps: float = 1e-06, reduction: Literal['none', 'mean', 'sum'] = 'none') -> mlx.core.array
mlx.nn.mish(x: mlx.core.array) -> mlx.core.array
mlx.nn.prelu(x: mlx.core.array, alpha: mlx.core.array) -> mlx.core.array
mlx.nn.relu(x)
mlx.nn.relu6(x)
mlx.nn.selu(x)
mlx.nn.silu(x)
mlx.nn.softmax(x, axis=-1)
mlx.nn.softplus(x)
mlx.nn.softsign(x)
mlx.nn.step(x: mlx.core.array, threshold: float = 0.0)
mlx.nn.tanh(x)
mlx.nn.utils.value_and_grad(model: 'mlx.nn.Module', fn: Callable)
mlx.nn.value_and_grad(model: 'mlx.nn.Module', fn: Callable)
mlx.optimizers.AdaDelta(learning_rate: float, rho: float = 0.9, eps: float = 1e-06)
mlx.optimizers.Adagrad(learning_rate: float, eps: float = 1e-08)
mlx.optimizers.Adam(learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-08)
mlx.optimizers.AdamW(learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-08, weight_decay: float = 0.01)
mlx.optimizers.Adamax(learning_rate: float, betas: List[float] = [0.9, 0.999], eps: float = 1e-08)
mlx.optimizers.Lion(learning_rate: float, betas: List[float] = [0.9, 0.99], weight_decay: float = 0.0)
mlx.optimizers.Optimizer()
mlx.optimizers.RMSprop(learning_rate: float, alpha: float = 0.99, eps: float = 1e-08)
mlx.optimizers.SGD(learning_rate: float, momentum: float = 0.0, weight_decay: float = 0.0, dampening: float = 0.0, nesterov: bool = False)
mlx.optimizers.tree_map(fn, tree, *rest, is_leaf=None)
mlx.utils.tree_flatten(tree, prefix='', is_leaf=None)
mlx.utils.tree_map(fn, tree, *rest, is_leaf=None)
mlx.utils.tree_unflatten(tree)