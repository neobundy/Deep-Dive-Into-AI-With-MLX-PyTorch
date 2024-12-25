
## mlx._reprlib_fix.FixedRepr
                Only route python array instances to repr_array.
                
## mlx.core.DeviceType
                Members:

cpu

gpu
                
## mlx.core.Dtype
                An object to hold the type of a :class:`array`.

See the :ref:`list of types <data_types>` for more details
on available data types.
                
## mlx.core._ArrayAt
                A helper object to apply updates at specific indices.
                
## mlx.core._ArrayIterator
                A helper object to iterate over the 1st dimension of an array.
                
## mlx.core.array
                An N-dimensional array object.
                
## mlx.extension.CMakeBuild
                Setuptools internal actions are organized using a *command design pattern*.
This means that each action (or group of closely related actions) executed during
the build should be implemented as a ``Command`` subclass.

These commands are abstractions and do not necessarily correspond to a command that
can (or should) be executed via a terminal, in a CLI fashion (although historically
they would).

When creating a new command from scratch, custom defined classes **SHOULD** inherit
from ``setuptools.Command`` and implement a few mandatory methods.
Between these mandatory methods, are listed:

.. method:: initialize_options(self)

    Set or (reset) all options/attributes/caches used by the command
    to their default values. Note that these values may be overwritten during
    the build.

.. method:: finalize_options(self)

    Set final values for all options/attributes used by the command.
    Most of the time, each option/attribute/cache should only be set if it does not
    have any value yet (e.g. ``if self.attr is None: self.attr = val``).

.. method:: run(self)

    Execute the actions intended by the command.
    (Side effects **SHOULD** only take place when ``run`` is executed,
    for example, creating new files or writing to the terminal output).

A useful analogy for command classes is to think of them as subroutines with local
variables called "options".  The options are "declared" in ``initialize_options()``
and "defined" (given their final values, aka "finalized") in ``finalize_options()``,
both of which must be defined by every command class. The "body" of the subroutine,
(where it does all the work) is the ``run()`` method.
Between ``initialize_options()`` and ``finalize_options()``, ``setuptools`` may set
the values for options/attributes based on user's input (or circumstance),
which means that the implementation should be careful to not overwrite values in
``finalize_options`` unless necessary.

Please note that other commands (or other parts of setuptools) may also overwrite
the values of the command's options/attributes multiple times during the build
process.
Therefore it is important to consistently implement ``initialize_options()`` and
``finalize_options()``. For example, all derived attributes (or attributes that
depend on the value of other attributes) **SHOULD** be recomputed in
``finalize_options``.

When overwriting existing commands, custom defined classes **MUST** abide by the
same APIs implemented by the original class. They also **SHOULD** inherit from the
original class.
                
## mlx.extension.CMakeExtension
                Describes a single extension module.

This means that all source files will be compiled into a single binary file
``<module path>.<suffix>`` (with ``<module path>`` derived from ``name`` and
``<suffix>`` defined by one of the values in
``importlib.machinery.EXTENSION_SUFFIXES``).

In the case ``.pyx`` files are passed as ``sources and`` ``Cython`` is **not**
installed in the build environment, ``setuptools`` may also try to look for the
equivalent ``.cpp`` or ``.c`` files.

:arg str name:
  the full name of the extension, including any packages -- ie.
  *not* a filename or pathname, but Python dotted name

:arg list[str] sources:
  list of source filenames, relative to the distribution root
  (where the setup script lives), in Unix form (slash-separated)
  for portability.  Source files may be C, C++, SWIG (.i),
  platform-specific resource files, or whatever else is recognized
  by the "build_ext" command as source for a Python extension.

:keyword list[str] include_dirs:
  list of directories to search for C/C++ header files (in Unix
  form for portability)

:keyword list[tuple[str, str|None]] define_macros:
  list of macros to define; each macro is defined using a 2-tuple:
  the first item corresponding to the name of the macro and the second
  item either a string with its value or None to
  define it without a particular value (equivalent of "#define
  FOO" in source or -DFOO on Unix C compiler command line)

:keyword list[str] undef_macros:
  list of macros to undefine explicitly

:keyword list[str] library_dirs:
  list of directories to search for C/C++ libraries at link time

:keyword list[str] libraries:
  list of library names (not filenames or paths) to link against

:keyword list[str] runtime_library_dirs:
  list of directories to search for C/C++ libraries at run time
  (for shared extensions, this is when the extension is loaded).
  Setting this will cause an exception during build on Windows
  platforms.

:keyword list[str] extra_objects:
  list of extra files to link with (eg. object files not implied
  by 'sources', static library that must be explicitly specified,
  binary resource files, etc.)

:keyword list[str] extra_compile_args:
  any extra platform- and compiler-specific information to use
  when compiling the source files in 'sources'.  For platforms and
  compilers where "command line" makes sense, this is typically a
  list of command-line arguments, but for other platforms it could
  be anything.

:keyword list[str] extra_link_args:
  any extra platform- and compiler-specific information to use
  when linking object files together to create the extension (or
  to create a new static Python interpreter).  Similar
  interpretation as for 'extra_compile_args'.

:keyword list[str] export_symbols:
  list of symbols to be exported from a shared extension.  Not
  used on all platforms, and not generally necessary for Python
  extensions, which typically export exactly one symbol: "init" +
  extension_name.

:keyword list[str] swig_opts:
  any extra options to pass to SWIG if a source file has the .i
  extension.

:keyword list[str] depends:
  list of files that the extension depends on

:keyword str language:
  extension language (i.e. "c", "c++", "objc"). Will be detected
  from the source extensions if not provided.

:keyword bool optional:
  specifies that a build failure in the extension should not abort the
  build process, but simply not install the failing extension.

:keyword bool py_limited_api:
  opt-in flag for the usage of :doc:`Python's limited API <python:c-api/stable>`.

:raises setuptools.errors.PlatformError: if 'runtime_library_dirs' is
  specified on Windows. (since v63)
                
## mlx.extension.Extension
                Describes a single extension module.

This means that all source files will be compiled into a single binary file
``<module path>.<suffix>`` (with ``<module path>`` derived from ``name`` and
``<suffix>`` defined by one of the values in
``importlib.machinery.EXTENSION_SUFFIXES``).

In the case ``.pyx`` files are passed as ``sources and`` ``Cython`` is **not**
installed in the build environment, ``setuptools`` may also try to look for the
equivalent ``.cpp`` or ``.c`` files.

:arg str name:
  the full name of the extension, including any packages -- ie.
  *not* a filename or pathname, but Python dotted name

:arg list[str] sources:
  list of source filenames, relative to the distribution root
  (where the setup script lives), in Unix form (slash-separated)
  for portability.  Source files may be C, C++, SWIG (.i),
  platform-specific resource files, or whatever else is recognized
  by the "build_ext" command as source for a Python extension.

:keyword list[str] include_dirs:
  list of directories to search for C/C++ header files (in Unix
  form for portability)

:keyword list[tuple[str, str|None]] define_macros:
  list of macros to define; each macro is defined using a 2-tuple:
  the first item corresponding to the name of the macro and the second
  item either a string with its value or None to
  define it without a particular value (equivalent of "#define
  FOO" in source or -DFOO on Unix C compiler command line)

:keyword list[str] undef_macros:
  list of macros to undefine explicitly

:keyword list[str] library_dirs:
  list of directories to search for C/C++ libraries at link time

:keyword list[str] libraries:
  list of library names (not filenames or paths) to link against

:keyword list[str] runtime_library_dirs:
  list of directories to search for C/C++ libraries at run time
  (for shared extensions, this is when the extension is loaded).
  Setting this will cause an exception during build on Windows
  platforms.

:keyword list[str] extra_objects:
  list of extra files to link with (eg. object files not implied
  by 'sources', static library that must be explicitly specified,
  binary resource files, etc.)

:keyword list[str] extra_compile_args:
  any extra platform- and compiler-specific information to use
  when compiling the source files in 'sources'.  For platforms and
  compilers where "command line" makes sense, this is typically a
  list of command-line arguments, but for other platforms it could
  be anything.

:keyword list[str] extra_link_args:
  any extra platform- and compiler-specific information to use
  when linking object files together to create the extension (or
  to create a new static Python interpreter).  Similar
  interpretation as for 'extra_compile_args'.

:keyword list[str] export_symbols:
  list of symbols to be exported from a shared extension.  Not
  used on all platforms, and not generally necessary for Python
  extensions, which typically export exactly one symbol: "init" +
  extension_name.

:keyword list[str] swig_opts:
  any extra options to pass to SWIG if a source file has the .i
  extension.

:keyword list[str] depends:
  list of files that the extension depends on

:keyword str language:
  extension language (i.e. "c", "c++", "objc"). Will be detected
  from the source extensions if not provided.

:keyword bool optional:
  specifies that a build failure in the extension should not abort the
  build process, but simply not install the failing extension.

:keyword bool py_limited_api:
  opt-in flag for the usage of :doc:`Python's limited API <python:c-api/stable>`.

:raises setuptools.errors.PlatformError: if 'runtime_library_dirs' is
  specified on Windows. (since v63)
                
## mlx.extension.Path
                PurePath subclass that can make system calls.

Path represents a filesystem path but unlike PurePath, also offers
methods to do system calls on path objects. Depending on your system,
instantiating a Path will return either a PosixPath or a WindowsPath
object. You can also instantiate a PosixPath or WindowsPath directly,
but cannot instantiate a WindowsPath on a POSIX system or vice versa.
                
## mlx.extension.build_ext
                Setuptools internal actions are organized using a *command design pattern*.
This means that each action (or group of closely related actions) executed during
the build should be implemented as a ``Command`` subclass.

These commands are abstractions and do not necessarily correspond to a command that
can (or should) be executed via a terminal, in a CLI fashion (although historically
they would).

When creating a new command from scratch, custom defined classes **SHOULD** inherit
from ``setuptools.Command`` and implement a few mandatory methods.
Between these mandatory methods, are listed:

.. method:: initialize_options(self)

    Set or (reset) all options/attributes/caches used by the command
    to their default values. Note that these values may be overwritten during
    the build.

.. method:: finalize_options(self)

    Set final values for all options/attributes used by the command.
    Most of the time, each option/attribute/cache should only be set if it does not
    have any value yet (e.g. ``if self.attr is None: self.attr = val``).

.. method:: run(self)

    Execute the actions intended by the command.
    (Side effects **SHOULD** only take place when ``run`` is executed,
    for example, creating new files or writing to the terminal output).

A useful analogy for command classes is to think of them as subroutines with local
variables called "options".  The options are "declared" in ``initialize_options()``
and "defined" (given their final values, aka "finalized") in ``finalize_options()``,
both of which must be defined by every command class. The "body" of the subroutine,
(where it does all the work) is the ``run()`` method.
Between ``initialize_options()`` and ``finalize_options()``, ``setuptools`` may set
the values for options/attributes based on user's input (or circumstance),
which means that the implementation should be careful to not overwrite values in
``finalize_options`` unless necessary.

Please note that other commands (or other parts of setuptools) may also overwrite
the values of the command's options/attributes multiple times during the build
process.
Therefore it is important to consistently implement ``initialize_options()`` and
``finalize_options()``. For example, all derived attributes (or attributes that
depend on the value of other attributes) **SHOULD** be recomputed in
``finalize_options``.

When overwriting existing commands, custom defined classes **MUST** abide by the
same APIs implemented by the original class. They also **SHOULD** inherit from the
original class.
                
## mlx.extension.find_namespace_packages
                Return a list of all Python items (packages or modules, depending on
the finder implementation) found within directory 'where'.

'where' is the root directory which will be searched.
It should be supplied as a "cross-platform" (i.e. URL-style) path;
it will be converted to the appropriate local path syntax.

'exclude' is a sequence of names to exclude; '*' can be used
as a wildcard in the names.
When finding packages, 'foo.*' will exclude all subpackages of 'foo'
(but not 'foo' itself).

'include' is a sequence of names to include.
If it's specified, only the named items will be included.
If it's not specified, all found items will be included.
'include' can contain shell style wildcard patterns just like
'exclude'.
                
## mlx.extension.setup
                The gateway to the Distutils: do everything your setup script needs
to do, in a highly flexible and user-driven way.  Briefly: create a
Distribution instance; find and parse config files; parse the command
line; run each Distutils command found there, customized by the options
supplied to 'setup()' (as keyword arguments), in config files, and on
the command line.

The Distribution instance might be an instance of a class supplied via
the 'distclass' keyword argument to 'setup'; if no such class is
supplied, then the Distribution class (in dist.py) is instantiated.
All other arguments to 'setup' (except for 'cmdclass') are used to set
attributes of the Distribution instance.

The 'cmdclass' argument, if supplied, is a dictionary mapping command
names to command classes.  Each command encountered on the command line
will be turned into a command class, which is in turn instantiated; any
class found in 'cmdclass' is used in place of the default, which is
(for command 'foo_bar') class 'foo_bar' in module
'distutils.command.foo_bar'.  The command class must provide a
'user_options' attribute which is a list of option specifiers for
'distutils.fancy_getopt'.  Any command-line options between the current
and the next command are used to set attributes of the current command
object.

When the entire command-line has been successfully parsed, calls the
'run()' method on each command object in turn.  This method will be
driven entirely by the Distribution object (which each command object
has a reference to, thanks to its constructor), and the
command-specific options that became attributes of each command
object.
                
## mlx.nn.ALiBi
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.BatchNorm
                Applies Batch Normalization over a 2D or 3D input.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
batch, ``C`` is the number of features or channels, and ``L`` is the
sequence length. The output has the same shape as the input. For
four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
the height and width respectively.

For more information on Batch Normalization, see the original paper `Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

Args:
    num_features (int): The feature dimension to normalize over.
    eps (float, optional): A small additive constant for numerical
        stability. Default: ``1e-5``.
    momentum (float, optional): The momentum for updating the running
        mean and variance. Default: ``0.1``.
    affine (bool, optional): If ``True``, apply a learned affine
        transformation after the normalization. Default: ``True``.
    track_running_stats (bool, optional): If ``True``, track the
        running mean and variance. Default: ``True``.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((5, 4))
    >>> bn = nn.BatchNorm(num_features=4, affine=True)
    >>> output = bn(x)
                
## mlx.nn.Bilinear
                Applies a bilinear transformation to the inputs.

Concretely:

.. math::

    y_i = x_1^\top W_i x_2 + b_i

where:
:math:`W` has shape ``[output_dims, input1_dims, input2_dims]``, :math:`b` has shape ``[output_dims ]``,
and :math:`i` indexes the output dimension.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_1}}` and :math:`D_1` is ``input1_dims``.

Args:
    input1_dims (int): The dimensionality of the input1 features
    input2_dims (int): The dimensionality of the input2 features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.CELU
                Applies the Continuously Differentiable Exponential Linear Unit.
    Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
    element wise.

See :func:`celu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
                
## mlx.nn.Conv1d
                Applies a 1-dimensional convolution over the multi-channel input sequence.

The channels are expected to be last i.e. the input shape should be ``NLC`` where:
    - ``N`` is the batch dimension
    - ``L`` is the sequence length
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    kernel_size (int): The size of the convolution filters
    stride (int, optional): The stride when applying the filter.
        Default: 1.
    padding (int, optional): How many positions to 0-pad the input with.
        Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the output.
        Default: ``True``
                
## mlx.nn.Conv2d
                Applies a 2-dimensional convolution over the multi-channel input image.

The channels are expected to be last i.e. the input shape should be ``NHWC`` where:
    - ``N`` is the batch dimension
    - ``H`` is the input image height
    - ``W`` is the input image width
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels.
    out_channels (int): The number of output channels.
    kernel_size (int or tuple): The size of the convolution filters.
    stride (int or tuple, optional): The size of the stride when
        applying the filter. Default: 1.
    padding (int or tuple, optional): How many positions to 0-pad
        the input with. Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the
        output. Default: ``True``
                
## mlx.nn.Dropout
                Randomly zero a portion of the elements during training.

The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
:math:`p` is the probability of zeroing an element. This is done so the
expected value of a given element will remain the same.

Args:
    p (float): The probability to zero an element
                
## mlx.nn.Dropout2d
                Apply 2D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e. the input shape should be
``NWHC`` or ``WHC`` where:``N`` is the batch dimension,``H`` is the input
image height,``W`` is the input image width, and``C`` is the number of
input channels

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
beneficial for early convolution layers where adjacent pixels are
correlated. In such case, traditional dropout may not effectively
regularize activations. For more details, see [1].

[1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
Efficient Object Localization Using Convolutional Networks. CVPR 2015.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.Dropout3d
                Apply 3D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e., the input shape should be
`NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
`H` is the input image height, `W` is the input image width, and `C` is
the number of input channels.

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
often beneficial for convolutional layers processing 3D data, like in
medical imaging or video processing.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.ELU
                Applies the Exponential Linear Unit.
    Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.

See :func:`elu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
                
## mlx.nn.Embedding
                Implements a simple lookup table that maps each input integer to a
high-dimensional vector.

Typically used to embed discrete tokens for processing by neural networks.

Args:
    num_embeddings (int): How many possible discrete tokens can we embed.
                          Usually called the vocabulary size.
    dims (int): The dimensionality of the embeddings.
                
## mlx.nn.GELU
                Applies the Gaussian Error Linear Units.

.. math::
    \textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

However, if ``approx`` is set to 'precise' or 'fast' it applies

.. math::
    \textrm{GELUApprox}(x) &= x * \sigma\left(1.60033 * x \left(1 + 0.0433603 * x^2\right)\right) \\
    \textrm{GELUFast}(x) &= x * \sigma\left(1.773 * x\right)

respectively.

See :func:`gelu`, :func:`gelu_approx` and :func:`gelu_fast_approx` for the
functional equivalents and information regarding error bounds.

Args:
    approx ('none' | 'precise' | 'fast'): Which approximation to gelu to use if any.
                
## mlx.nn.GLU
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.GroupNorm
                Applies Group Normalization [1] to the inputs.

Computes the same normalization as layer norm, namely

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. However, the mean and
variance are computed over the spatial dimensions and each group of
features. In particular, the input is split into num_groups across the
feature dimension.

The feature dimension is assumed to be the last dimension and the dimensions
that precede it (except the first) are considered the spatial dimensions.

[1]: https://arxiv.org/abs/1803.08494

Args:
    num_groups (int): Number of groups to separate the features into
    dims (int): The feature dimensions of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization.
    pytorch_compatible (bool): If True perform the group normalization in
        the same order/grouping as PyTorch.
                
## mlx.nn.Hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.Identity
                A placeholder identity operator that is argument-insensitive.

Args:
    args: any argument (unused)
    kwargs: any keyword argument (unused)
                
## mlx.nn.InstanceNorm
                Applies instance normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. Both are of size :attr:`dims`,
if :attr:`affine` is ``True``.

Args:
    dims (int): The number of features of the input.
    eps (float): A value added to the denominator for numerical stability. Default: ``1e-5``.
    affine (bool): Default: ``False``.

Shape:
  - Input: :math:`(..., C)` where :math:`C` is equal to :attr:`dims`.
  - Output: Same shape as the input.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((8, 4, 4, 16))
    >>> inorm = nn.InstanceNorm(dims=16)
    >>> output = inorm(x)

References:
    [1]: https://arxiv.org/abs/1607.08022
                
## mlx.nn.LayerNorm
                Applies layer normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

[1]: https://arxiv.org/abs/1607.06450

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization
                
## mlx.nn.LeakyReLU
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.

Args:
    negative_slope: Controls the angle of the negative slope. Default: 1e-2.
                
## mlx.nn.Linear
                Applies an affine transformation to the input.

Concretely:

.. math::

    y = x W^\top + b

where:
where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.LogSigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.LogSoftmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.Mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.MultiHeadAttention
                Implements the scaled dot product attention with multiple heads.

Given inputs for queries, keys and values the ``MultiHeadAttention``
produces new values by aggregating information from the input values
according to the similarities of the input queries and keys.

All inputs as well as the output are linearly projected without biases by
default.

``MultiHeadAttention`` also takes an optional additive attention mask that
should be broadcastable with ``(batch, num_heads, # queries, # keys)``. The
mask should have ``-inf`` or very large negative numbers at the positions
that should *not* be attended to.

Args:
    dims (int): The model dimensions. This is also the default
        value for the queries, keys, values, and the output.
    num_heads (int): The number of attention heads to use.
    query_input_dims (int, optional): The input dimensions of the queries.
        Default: ``dims``.
    key_input_dims (int, optional): The input dimensions of the keys.
        Default: ``dims``.
    value_input_dims (int, optional): The input dimensions of the values.
        Default: ``key_input_dims``.
    value_dims (int, optional): The dimensions of the values after the
        projection. Default: ``dims``.
    value_output_dims (int, optional): The dimensions the new values will
        be projected to. Default: ``dims``.
    bias (bool, optional): Whether or not to use a bias in the projections.
        Default: ``False``.
                
## mlx.nn.PReLU
                Applies the element-wise parametric ReLU.
    Applies :math:`\max(0, x) + a * \min(0, x)` element wise, where :math:`a`
    is an array.

See :func:`prelu`, for the functional equivalent.

Args:
    num_parameters: number of :math:`a` to learn. Default: 1
    init: the initial value of :math:`a`. Default: 0.25
                
## mlx.nn.QuantizedLinear
                Applies an affine transformation to the input using a quantized weight matrix.

It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
parameters are frozen and will not be included in any gradient computation
but this will probably change in the future.

QuantizedLinear also provides two useful classmethods to convert linear
layers to QuantizedLinear layers.

- :meth:`from_linear` returns a QuantizedLinear layer that applies the same
  linear transformation up to the quantization error.
- :meth:`quantize_module` swaps all the linear layers of the passed module
  with QuantizedLinear ones.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will not use
        a bias. (default: True).
    group_size (int, optional): The group size to use for the quantized
        weight. See :func:`~mlx.core.quantize`. (default: 64)
    bits (int, optional): The bit width to use for the quantized weight.
        See :func:`~mlx.core.quantize`. (default: 4)
                
## mlx.nn.RMSNorm
                Applies Root Mean Square normalization [1] to the inputs.

Computes

..  math::

    y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

where :math:`\gamma` is a learned per feature dimension parameter initialized at
1.

[1]: https://arxiv.org/abs/1910.07467

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
                
## mlx.nn.ReLU
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.ReLU6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.RoPE
                Implements the rotary positional encoding.

The traditional implementation rotates consecutive pairs of elements in the
feature dimension while the default implementation rotates pairs with
stride half the feature dimensions for efficiency.

For more details see `RoFormer: Enhanced Transformer with Rotary Position
Embedding <https://arxiv.org/abs/2104.09864>`_.

Args:
    dims (int): The feature dimensions to be rotated. If the input feature
        is larger than dims then the rest is left unchanged.
    traditional (bool, optional): If set to True choose the traditional
        implementation which is slightly less efficient. Default: ``False``.
    base (float, optional): The base used to compute angular frequency for
        each dimension in the positional encodings. Default: ``10000``.
    scale (float, optional): The scale used to scale the positions. Default: ``1.0``.

Attributes:
    _cos_sin_theta_key (tuple): Cached key for the precomputed cosine and sine values.
    _cos_sin_theta_value (tuple): Cached cosine and sine values.
                
## mlx.nn.SELU
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.Sequential
                A layer that calls the passed callables in order.

We can pass either modules or plain callables to the Sequential module. If
our functions have learnable parameters they should be implemented as
``nn.Module`` instances.

Args:
    modules (tuple of Callables): The modules to call in order
                
## mlx.nn.SiLU
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.Sigmoid
                sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

Element-wise logistic sigmoid.

The logistic sigmoid function is:

.. math::
  \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

Args:
    a (array): Input array.

Returns:
    array: The logistic sigmoid of ``a``.
                
## mlx.nn.SinusoidalPositionalEncoding
                Implements sinusoidal positional encoding.

For more details see the paper `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

Args:
    dims (int): The dimensionality of the resulting positional embeddings.
    min_freq (float, optional): The minimum frequency expected. Default:
        ``0.0001``.
    max_freq (float, optional): The maximum frequency expected. Default:
        ``1``.
    scale (float, optional): A multiplicative scale for the embeddings.
        Default: ``sqrt(dims//2)``.
    cos_first (bool, optional): If ``True`` embed using ``[cos(x); sin(x)]``
        instead of the reverse. Default: ``False``.
    full_turns (bool, optional): If ``True`` multiply the frequencies with
        :math:`2\pi`. Default: ``False``.
                
## mlx.nn.Softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.Softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.Softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.Step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.Tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.Transformer
                Implements a standard Transformer model.

The implementation is based on `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

The Transformer model contains an encoder and a decoder. The encoder
processes the input sequence and the decoder generates the output sequence.
The interaction between encoder and decoder happens through the attention
mechanism.

Args:
    dims (int, optional): The number of expected features in the
        encoder/decoder inputs. Default: ``512``.
    num_heads (int, optional): The number of attention heads. Default:
        ``8``.
    num_encoder_layers (int, optional): The number of encoder layers in the
        Transformer encoder. Default: ``6``.
    num_decoder_layers (int, optional): The number of decoder layers in the
        Transformer decoder. Default: ``6``.
    mlp_dims (int, optional): The hidden dimension of the MLP block in each
        Transformer layer. Defaults to ``4*dims`` if not provided. Default:
        ``None``.
    dropout (float, optional): The dropout value for the Transformer
        encoder and decoder. Dropout is used after each attention layer and
        the activation in the MLP layer. Default: ``0.0``.
    activation (function, optional): the activation function for the MLP
        hidden layer. Default: :func:`mlx.nn.relu`.
    custom_encoder (nn.Module, optional): A custom encoder to replace the
        standard Transformer encoder. Default: ``None``.
    custom_decoder (nn.Module, optional): A custom decoder to replace the
        standard Transformer decoder. Default: ``None``.
    norm_first (bool, optional): if ``True``, encoder and decoder layers
        will perform layer normalization before attention and MLP
        operations, otherwise after. Default: ``False``.
                
## mlx.nn.TransformerEncoder
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.TransformerEncoderLayer
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.activations.Any
                Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
                
## mlx.nn.layers.activations.CELU
                Applies the Continuously Differentiable Exponential Linear Unit.
    Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
    element wise.

See :func:`celu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
                
## mlx.nn.layers.activations.ELU
                Applies the Exponential Linear Unit.
    Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.

See :func:`elu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
                
## mlx.nn.layers.activations.GELU
                Applies the Gaussian Error Linear Units.

.. math::
    \textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

However, if ``approx`` is set to 'precise' or 'fast' it applies

.. math::
    \textrm{GELUApprox}(x) &= x * \sigma\left(1.60033 * x \left(1 + 0.0433603 * x^2\right)\right) \\
    \textrm{GELUFast}(x) &= x * \sigma\left(1.773 * x\right)

respectively.

See :func:`gelu`, :func:`gelu_approx` and :func:`gelu_fast_approx` for the
functional equivalents and information regarding error bounds.

Args:
    approx ('none' | 'precise' | 'fast'): Which approximation to gelu to use if any.
                
## mlx.nn.layers.activations.GLU
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.layers.activations.Hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.layers.activations.LeakyReLU
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.

Args:
    negative_slope: Controls the angle of the negative slope. Default: 1e-2.
                
## mlx.nn.layers.activations.LogSigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.layers.activations.LogSoftmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.layers.activations.Mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.layers.activations.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.activations.PReLU
                Applies the element-wise parametric ReLU.
    Applies :math:`\max(0, x) + a * \min(0, x)` element wise, where :math:`a`
    is an array.

See :func:`prelu`, for the functional equivalent.

Args:
    num_parameters: number of :math:`a` to learn. Default: 1
    init: the initial value of :math:`a`. Default: 0.25
                
## mlx.nn.layers.activations.ReLU
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.layers.activations.ReLU6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.layers.activations.SELU
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.layers.activations.SiLU
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.layers.activations.Sigmoid
                sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

Element-wise logistic sigmoid.

The logistic sigmoid function is:

.. math::
  \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

Args:
    a (array): Input array.

Returns:
    array: The logistic sigmoid of ``a``.
                
## mlx.nn.layers.activations.Softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.layers.activations.Softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.layers.activations.Softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.layers.activations.Step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.layers.activations.Tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.layers.activations.celu
                Applies the Continuously Differentiable Exponential Linear Unit.

Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
element wise.
                
## mlx.nn.layers.activations.elu
                Applies the Exponential Linear Unit.

Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.
                
## mlx.nn.layers.activations.gelu
                Applies the Gaussian Error Linear Units function.

.. math::
    \\textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

See also :func:`gelu_approx` and :func:`gelu_fast_approx` for faster
approximations.
                
## mlx.nn.layers.activations.gelu_approx
                An approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.0003` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.60033 x \left(1 + 0.0433603 x^2\right)\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.layers.activations.gelu_fast_approx
                A fast approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.015` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.773 x\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.layers.activations.glu
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.layers.activations.hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.layers.activations.leaky_relu
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.
                
## mlx.nn.layers.activations.log_sigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.layers.activations.log_softmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.layers.activations.mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.layers.activations.prelu
                Applies the element-wise parametric ReLU.

.. math::
    \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

where :math:`a` is an array.
                
## mlx.nn.layers.activations.relu
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.layers.activations.relu6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.layers.activations.selu
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.layers.activations.sigmoid
                Applies the element-wise function:

.. math::
    \text{Sigmoid}(x) = \sigma(x) = \frac{1}{1 + \exp(-x)}
                
## mlx.nn.layers.activations.silu
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.layers.activations.softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.layers.activations.softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.layers.activations.softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.layers.activations.step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.layers.activations.tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.layers.base.Any
                Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
                
## mlx.nn.layers.base.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.base.tree_flatten
                Flattens a python tree to a list of key, value tuples.

The keys are using the dot notation to define trees of arbitrary depth and
complexity.

.. code-block:: python

    from mlx.utils import tree_flatten

    print(tree_flatten([[[0]]]))
    # [("0.0.0", 0)]

    print(tree_flatten([[[0]]], ".hello"))
    # [("hello.0.0.0", 0)]

.. note::
   Dictionaries should have keys that are valid python identifiers.

Args:
    tree (Any): The python tree to be flattened.
    prefix (str): A prefix to use for the keys. The first character is
        always discarded.
    is_leaf (Callable): An optional callable that returns True if the
        passed object is considered a leaf or False otherwise.

Returns:
    List[Tuple[str, Any]]: The flat representation of the python tree.
                
## mlx.nn.layers.base.tree_unflatten
                Recreate a python tree from its flat representation.

.. code-block:: python

    from mlx.utils import tree_unflatten

    d = tree_unflatten([("hello.world", 42)])
    print(d)
    # {"hello": {"world": 42}}

Args:
    tree (List[Tuple[str, Any]]): The flat representation of a python tree.
                                  For instance as returned by :meth:`tree_flatten`.

Returns:
    A python tree.
                
## mlx.nn.celu
                Applies the Continuously Differentiable Exponential Linear Unit.

Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
element wise.
                
## mlx.nn.layers.containers.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.containers.Sequential
                A layer that calls the passed callables in order.

We can pass either modules or plain callables to the Sequential module. If
our functions have learnable parameters they should be implemented as
``nn.Module`` instances.

Args:
    modules (tuple of Callables): The modules to call in order
                
## mlx.nn.layers.convolution.Conv1d
                Applies a 1-dimensional convolution over the multi-channel input sequence.

The channels are expected to be last i.e. the input shape should be ``NLC`` where:
    - ``N`` is the batch dimension
    - ``L`` is the sequence length
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    kernel_size (int): The size of the convolution filters
    stride (int, optional): The stride when applying the filter.
        Default: 1.
    padding (int, optional): How many positions to 0-pad the input with.
        Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the output.
        Default: ``True``
                
## mlx.nn.layers.convolution.Conv2d
                Applies a 2-dimensional convolution over the multi-channel input image.

The channels are expected to be last i.e. the input shape should be ``NHWC`` where:
    - ``N`` is the batch dimension
    - ``H`` is the input image height
    - ``W`` is the input image width
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels.
    out_channels (int): The number of output channels.
    kernel_size (int or tuple): The size of the convolution filters.
    stride (int or tuple, optional): The size of the stride when
        applying the filter. Default: 1.
    padding (int or tuple, optional): How many positions to 0-pad
        the input with. Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the
        output. Default: ``True``
                
## mlx.nn.layers.convolution.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.dropout.Dropout
                Randomly zero a portion of the elements during training.

The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
:math:`p` is the probability of zeroing an element. This is done so the
expected value of a given element will remain the same.

Args:
    p (float): The probability to zero an element
                
## mlx.nn.layers.dropout.Dropout2d
                Apply 2D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e. the input shape should be
``NWHC`` or ``WHC`` where:``N`` is the batch dimension,``H`` is the input
image height,``W`` is the input image width, and``C`` is the number of
input channels

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
beneficial for early convolution layers where adjacent pixels are
correlated. In such case, traditional dropout may not effectively
regularize activations. For more details, see [1].

[1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
Efficient Object Localization Using Convolutional Networks. CVPR 2015.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.layers.dropout.Dropout3d
                Apply 3D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e., the input shape should be
`NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
`H` is the input image height, `W` is the input image width, and `C` is
the number of input channels.

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
often beneficial for convolutional layers processing 3D data, like in
medical imaging or video processing.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.layers.dropout.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.elu
                Applies the Exponential Linear Unit.

Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.
                
## mlx.nn.layers.embedding.Embedding
                Implements a simple lookup table that maps each input integer to a
high-dimensional vector.

Typically used to embed discrete tokens for processing by neural networks.

Args:
    num_embeddings (int): How many possible discrete tokens can we embed.
                          Usually called the vocabulary size.
    dims (int): The dimensionality of the embeddings.
                
## mlx.nn.layers.embedding.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.gelu
                Applies the Gaussian Error Linear Units function.

.. math::
    \\textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

See also :func:`gelu_approx` and :func:`gelu_fast_approx` for faster
approximations.
                
## mlx.nn.gelu_approx
                An approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.0003` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.60033 x \left(1 + 0.0433603 x^2\right)\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.gelu_fast_approx
                A fast approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.015` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.773 x\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.glu
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.layers.ALiBi
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.BatchNorm
                Applies Batch Normalization over a 2D or 3D input.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
batch, ``C`` is the number of features or channels, and ``L`` is the
sequence length. The output has the same shape as the input. For
four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
the height and width respectively.

For more information on Batch Normalization, see the original paper `Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

Args:
    num_features (int): The feature dimension to normalize over.
    eps (float, optional): A small additive constant for numerical
        stability. Default: ``1e-5``.
    momentum (float, optional): The momentum for updating the running
        mean and variance. Default: ``0.1``.
    affine (bool, optional): If ``True``, apply a learned affine
        transformation after the normalization. Default: ``True``.
    track_running_stats (bool, optional): If ``True``, track the
        running mean and variance. Default: ``True``.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((5, 4))
    >>> bn = nn.BatchNorm(num_features=4, affine=True)
    >>> output = bn(x)
                
## mlx.nn.layers.Bilinear
                Applies a bilinear transformation to the inputs.

Concretely:

.. math::

    y_i = x_1^\top W_i x_2 + b_i

where:
:math:`W` has shape ``[output_dims, input1_dims, input2_dims]``, :math:`b` has shape ``[output_dims ]``,
and :math:`i` indexes the output dimension.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_1}}` and :math:`D_1` is ``input1_dims``.

Args:
    input1_dims (int): The dimensionality of the input1 features
    input2_dims (int): The dimensionality of the input2 features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.CELU
                Applies the Continuously Differentiable Exponential Linear Unit.
    Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
    element wise.

See :func:`celu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the CELU formulation. Default: 1.0
                
## mlx.nn.layers.Conv1d
                Applies a 1-dimensional convolution over the multi-channel input sequence.

The channels are expected to be last i.e. the input shape should be ``NLC`` where:
    - ``N`` is the batch dimension
    - ``L`` is the sequence length
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels
    out_channels (int): The number of output channels
    kernel_size (int): The size of the convolution filters
    stride (int, optional): The stride when applying the filter.
        Default: 1.
    padding (int, optional): How many positions to 0-pad the input with.
        Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the output.
        Default: ``True``
                
## mlx.nn.layers.Conv2d
                Applies a 2-dimensional convolution over the multi-channel input image.

The channels are expected to be last i.e. the input shape should be ``NHWC`` where:
    - ``N`` is the batch dimension
    - ``H`` is the input image height
    - ``W`` is the input image width
    - ``C`` is the number of input channels

Args:
    in_channels (int): The number of input channels.
    out_channels (int): The number of output channels.
    kernel_size (int or tuple): The size of the convolution filters.
    stride (int or tuple, optional): The size of the stride when
        applying the filter. Default: 1.
    padding (int or tuple, optional): How many positions to 0-pad
        the input with. Default: 0.
    bias (bool, optional): If ``True`` add a learnable bias to the
        output. Default: ``True``
                
## mlx.nn.layers.Dropout
                Randomly zero a portion of the elements during training.

The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
:math:`p` is the probability of zeroing an element. This is done so the
expected value of a given element will remain the same.

Args:
    p (float): The probability to zero an element
                
## mlx.nn.layers.Dropout2d
                Apply 2D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e. the input shape should be
``NWHC`` or ``WHC`` where:``N`` is the batch dimension,``H`` is the input
image height,``W`` is the input image width, and``C`` is the number of
input channels

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
beneficial for early convolution layers where adjacent pixels are
correlated. In such case, traditional dropout may not effectively
regularize activations. For more details, see [1].

[1]: Thompson, J., Goroshin, R., Jain, A., LeCun, Y. and Bregler C., 2015.
Efficient Object Localization Using Convolutional Networks. CVPR 2015.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.layers.Dropout3d
                Apply 3D channel-wise dropout during training.

Randomly zero out entire channels independently with probability :math:`p`.
This layer expects the channels to be last, i.e., the input shape should be
`NDHWC` or `DHWC` where: `N` is the batch dimension, `D` is the depth,
`H` is the input image height, `W` is the input image width, and `C` is
the number of input channels.

The remaining channels are scaled by :math:`\frac{1}{1-p}` to
maintain the expected value of each element. Unlike traditional dropout,
which zeros individual entries, this layer zeros entire channels. This is
often beneficial for convolutional layers processing 3D data, like in
medical imaging or video processing.

Args:
    p (float): Probability of zeroing a channel during training.
                
## mlx.nn.layers.ELU
                Applies the Exponential Linear Unit.
    Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.

See :func:`elu`, for the functional equivalent.

Args:
    alpha: the :math:`\alpha` value for the ELU formulation. Default: 1.0
                
## mlx.nn.layers.Embedding
                Implements a simple lookup table that maps each input integer to a
high-dimensional vector.

Typically used to embed discrete tokens for processing by neural networks.

Args:
    num_embeddings (int): How many possible discrete tokens can we embed.
                          Usually called the vocabulary size.
    dims (int): The dimensionality of the embeddings.
                
## mlx.nn.layers.GELU
                Applies the Gaussian Error Linear Units.

.. math::
    \textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

However, if ``approx`` is set to 'precise' or 'fast' it applies

.. math::
    \textrm{GELUApprox}(x) &= x * \sigma\left(1.60033 * x \left(1 + 0.0433603 * x^2\right)\right) \\
    \textrm{GELUFast}(x) &= x * \sigma\left(1.773 * x\right)

respectively.

See :func:`gelu`, :func:`gelu_approx` and :func:`gelu_fast_approx` for the
functional equivalents and information regarding error bounds.

Args:
    approx ('none' | 'precise' | 'fast'): Which approximation to gelu to use if any.
                
## mlx.nn.layers.GLU
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.layers.GroupNorm
                Applies Group Normalization [1] to the inputs.

Computes the same normalization as layer norm, namely

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. However, the mean and
variance are computed over the spatial dimensions and each group of
features. In particular, the input is split into num_groups across the
feature dimension.

The feature dimension is assumed to be the last dimension and the dimensions
that precede it (except the first) are considered the spatial dimensions.

[1]: https://arxiv.org/abs/1803.08494

Args:
    num_groups (int): Number of groups to separate the features into
    dims (int): The feature dimensions of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization.
    pytorch_compatible (bool): If True perform the group normalization in
        the same order/grouping as PyTorch.
                
## mlx.nn.layers.Hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.layers.Identity
                A placeholder identity operator that is argument-insensitive.

Args:
    args: any argument (unused)
    kwargs: any keyword argument (unused)
                
## mlx.nn.layers.InstanceNorm
                Applies instance normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. Both are of size :attr:`dims`,
if :attr:`affine` is ``True``.

Args:
    dims (int): The number of features of the input.
    eps (float): A value added to the denominator for numerical stability. Default: ``1e-5``.
    affine (bool): Default: ``False``.

Shape:
  - Input: :math:`(..., C)` where :math:`C` is equal to :attr:`dims`.
  - Output: Same shape as the input.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((8, 4, 4, 16))
    >>> inorm = nn.InstanceNorm(dims=16)
    >>> output = inorm(x)

References:
    [1]: https://arxiv.org/abs/1607.08022
                
## mlx.nn.layers.LayerNorm
                Applies layer normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

[1]: https://arxiv.org/abs/1607.06450

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization
                
## mlx.nn.layers.LeakyReLU
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.

Args:
    negative_slope: Controls the angle of the negative slope. Default: 1e-2.
                
## mlx.nn.layers.Linear
                Applies an affine transformation to the input.

Concretely:

.. math::

    y = x W^\top + b

where:
where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.LogSigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.layers.LogSoftmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.layers.Mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.layers.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.MultiHeadAttention
                Implements the scaled dot product attention with multiple heads.

Given inputs for queries, keys and values the ``MultiHeadAttention``
produces new values by aggregating information from the input values
according to the similarities of the input queries and keys.

All inputs as well as the output are linearly projected without biases by
default.

``MultiHeadAttention`` also takes an optional additive attention mask that
should be broadcastable with ``(batch, num_heads, # queries, # keys)``. The
mask should have ``-inf`` or very large negative numbers at the positions
that should *not* be attended to.

Args:
    dims (int): The model dimensions. This is also the default
        value for the queries, keys, values, and the output.
    num_heads (int): The number of attention heads to use.
    query_input_dims (int, optional): The input dimensions of the queries.
        Default: ``dims``.
    key_input_dims (int, optional): The input dimensions of the keys.
        Default: ``dims``.
    value_input_dims (int, optional): The input dimensions of the values.
        Default: ``key_input_dims``.
    value_dims (int, optional): The dimensions of the values after the
        projection. Default: ``dims``.
    value_output_dims (int, optional): The dimensions the new values will
        be projected to. Default: ``dims``.
    bias (bool, optional): Whether or not to use a bias in the projections.
        Default: ``False``.
                
## mlx.nn.layers.PReLU
                Applies the element-wise parametric ReLU.
    Applies :math:`\max(0, x) + a * \min(0, x)` element wise, where :math:`a`
    is an array.

See :func:`prelu`, for the functional equivalent.

Args:
    num_parameters: number of :math:`a` to learn. Default: 1
    init: the initial value of :math:`a`. Default: 0.25
                
## mlx.nn.layers.QuantizedLinear
                Applies an affine transformation to the input using a quantized weight matrix.

It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
parameters are frozen and will not be included in any gradient computation
but this will probably change in the future.

QuantizedLinear also provides two useful classmethods to convert linear
layers to QuantizedLinear layers.

- :meth:`from_linear` returns a QuantizedLinear layer that applies the same
  linear transformation up to the quantization error.
- :meth:`quantize_module` swaps all the linear layers of the passed module
  with QuantizedLinear ones.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will not use
        a bias. (default: True).
    group_size (int, optional): The group size to use for the quantized
        weight. See :func:`~mlx.core.quantize`. (default: 64)
    bits (int, optional): The bit width to use for the quantized weight.
        See :func:`~mlx.core.quantize`. (default: 4)
                
## mlx.nn.layers.RMSNorm
                Applies Root Mean Square normalization [1] to the inputs.

Computes

..  math::

    y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

where :math:`\gamma` is a learned per feature dimension parameter initialized at
1.

[1]: https://arxiv.org/abs/1910.07467

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
                
## mlx.nn.layers.ReLU
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.layers.ReLU6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.layers.RoPE
                Implements the rotary positional encoding.

The traditional implementation rotates consecutive pairs of elements in the
feature dimension while the default implementation rotates pairs with
stride half the feature dimensions for efficiency.

For more details see `RoFormer: Enhanced Transformer with Rotary Position
Embedding <https://arxiv.org/abs/2104.09864>`_.

Args:
    dims (int): The feature dimensions to be rotated. If the input feature
        is larger than dims then the rest is left unchanged.
    traditional (bool, optional): If set to True choose the traditional
        implementation which is slightly less efficient. Default: ``False``.
    base (float, optional): The base used to compute angular frequency for
        each dimension in the positional encodings. Default: ``10000``.
    scale (float, optional): The scale used to scale the positions. Default: ``1.0``.

Attributes:
    _cos_sin_theta_key (tuple): Cached key for the precomputed cosine and sine values.
    _cos_sin_theta_value (tuple): Cached cosine and sine values.
                
## mlx.nn.layers.SELU
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.layers.Sequential
                A layer that calls the passed callables in order.

We can pass either modules or plain callables to the Sequential module. If
our functions have learnable parameters they should be implemented as
``nn.Module`` instances.

Args:
    modules (tuple of Callables): The modules to call in order
                
## mlx.nn.layers.SiLU
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.layers.Sigmoid
                sigmoid(a: array, /, *, stream: Union[None, Stream, Device] = None) -> array

Element-wise logistic sigmoid.

The logistic sigmoid function is:

.. math::
  \mathrm{sigmoid}(x) = \frac{1}{1 + e^{-x}}

Args:
    a (array): Input array.

Returns:
    array: The logistic sigmoid of ``a``.
                
## mlx.nn.layers.SinusoidalPositionalEncoding
                Implements sinusoidal positional encoding.

For more details see the paper `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

Args:
    dims (int): The dimensionality of the resulting positional embeddings.
    min_freq (float, optional): The minimum frequency expected. Default:
        ``0.0001``.
    max_freq (float, optional): The maximum frequency expected. Default:
        ``1``.
    scale (float, optional): A multiplicative scale for the embeddings.
        Default: ``sqrt(dims//2)``.
    cos_first (bool, optional): If ``True`` embed using ``[cos(x); sin(x)]``
        instead of the reverse. Default: ``False``.
    full_turns (bool, optional): If ``True`` multiply the frequencies with
        :math:`2\pi`. Default: ``False``.
                
## mlx.nn.layers.Softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.layers.Softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.layers.Softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.layers.Step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.layers.Tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.layers.Transformer
                Implements a standard Transformer model.

The implementation is based on `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

The Transformer model contains an encoder and a decoder. The encoder
processes the input sequence and the decoder generates the output sequence.
The interaction between encoder and decoder happens through the attention
mechanism.

Args:
    dims (int, optional): The number of expected features in the
        encoder/decoder inputs. Default: ``512``.
    num_heads (int, optional): The number of attention heads. Default:
        ``8``.
    num_encoder_layers (int, optional): The number of encoder layers in the
        Transformer encoder. Default: ``6``.
    num_decoder_layers (int, optional): The number of decoder layers in the
        Transformer decoder. Default: ``6``.
    mlp_dims (int, optional): The hidden dimension of the MLP block in each
        Transformer layer. Defaults to ``4*dims`` if not provided. Default:
        ``None``.
    dropout (float, optional): The dropout value for the Transformer
        encoder and decoder. Dropout is used after each attention layer and
        the activation in the MLP layer. Default: ``0.0``.
    activation (function, optional): the activation function for the MLP
        hidden layer. Default: :func:`mlx.nn.relu`.
    custom_encoder (nn.Module, optional): A custom encoder to replace the
        standard Transformer encoder. Default: ``None``.
    custom_decoder (nn.Module, optional): A custom decoder to replace the
        standard Transformer decoder. Default: ``None``.
    norm_first (bool, optional): if ``True``, encoder and decoder layers
        will perform layer normalization before attention and MLP
        operations, otherwise after. Default: ``False``.
                
## mlx.nn.layers.TransformerEncoder
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.TransformerEncoderLayer
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.celu
                Applies the Continuously Differentiable Exponential Linear Unit.

Applies :math:`\max(0, x) + \min(0, \alpha * (\exp(x / \alpha) - 1))`
element wise.
                
## mlx.nn.layers.elu
                Applies the Exponential Linear Unit.

Simply ``mx.where(x > 0, x, alpha * (mx.exp(x) - 1))``.
                
## mlx.nn.layers.gelu
                Applies the Gaussian Error Linear Units function.

.. math::
    \\textrm{GELU}(x) = x * \Phi(x)

where :math:`\Phi(x)` is the Gaussian CDF.

See also :func:`gelu_approx` and :func:`gelu_fast_approx` for faster
approximations.
                
## mlx.nn.layers.gelu_approx
                An approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.0003` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.60033 x \left(1 + 0.0433603 x^2\right)\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.layers.gelu_fast_approx
                A fast approximation to Gaussian Error Linear Unit.

See :func:`gelu` for the exact computation.

This function approximates ``gelu`` with a maximum absolute error :math:`<
0.015` in the range :math:`[-6, 6]` using the following

.. math::

    x = x \sigma\left(1.773 x\right)

where :math:`\sigma(\cdot)` is the logistic sigmoid.
                
## mlx.nn.layers.glu
                Applies the gated linear unit function.

This function splits the ``axis`` dimension of the input into two halves
(:math:`a` and :math:`b`) and applies :math:`a * \sigma(b)`.

.. math::
    textrm{GLU}(x) = a * \sigma(b)

Args:
    axis (int): The dimension to split along. Default: ``-1``.
                
## mlx.nn.layers.hardswish
                Applies the hardswish function, element-wise.

.. math::
    \text{Hardswish}(x) = x * \min(\max(x + 3, 0), 6) / 6
                
## mlx.nn.layers.leaky_relu
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.
                
## mlx.nn.layers.linear.Any
                Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
                
## mlx.nn.layers.linear.Bilinear
                Applies a bilinear transformation to the inputs.

Concretely:

.. math::

    y_i = x_1^\top W_i x_2 + b_i

where:
:math:`W` has shape ``[output_dims, input1_dims, input2_dims]``, :math:`b` has shape ``[output_dims ]``,
and :math:`i` indexes the output dimension.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_1}}` and :math:`D_1` is ``input1_dims``.

Args:
    input1_dims (int): The dimensionality of the input1 features
    input2_dims (int): The dimensionality of the input2 features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.linear.Identity
                A placeholder identity operator that is argument-insensitive.

Args:
    args: any argument (unused)
    kwargs: any keyword argument (unused)
                
## mlx.nn.layers.linear.Linear
                Applies an affine transformation to the input.

Concretely:

.. math::

    y = x W^\top + b

where:
where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.linear.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.log_sigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.layers.log_softmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.layers.mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.layers.normalization.BatchNorm
                Applies Batch Normalization over a 2D or 3D input.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

The input shape is specified as ``NC`` or ``NLC``, where ``N`` is the
batch, ``C`` is the number of features or channels, and ``L`` is the
sequence length. The output has the same shape as the input. For
four-dimensional arrays, the shape is ``NHWC``, where ``H`` and ``W`` are
the height and width respectively.

For more information on Batch Normalization, see the original paper `Batch
Normalization: Accelerating Deep Network Training by Reducing Internal
Covariate Shift <https://arxiv.org/abs/1502.03167>`_.

Args:
    num_features (int): The feature dimension to normalize over.
    eps (float, optional): A small additive constant for numerical
        stability. Default: ``1e-5``.
    momentum (float, optional): The momentum for updating the running
        mean and variance. Default: ``0.1``.
    affine (bool, optional): If ``True``, apply a learned affine
        transformation after the normalization. Default: ``True``.
    track_running_stats (bool, optional): If ``True``, track the
        running mean and variance. Default: ``True``.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((5, 4))
    >>> bn = nn.BatchNorm(num_features=4, affine=True)
    >>> output = bn(x)
                
## mlx.nn.layers.normalization.GroupNorm
                Applies Group Normalization [1] to the inputs.

Computes the same normalization as layer norm, namely

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. However, the mean and
variance are computed over the spatial dimensions and each group of
features. In particular, the input is split into num_groups across the
feature dimension.

The feature dimension is assumed to be the last dimension and the dimensions
that precede it (except the first) are considered the spatial dimensions.

[1]: https://arxiv.org/abs/1803.08494

Args:
    num_groups (int): Number of groups to separate the features into
    dims (int): The feature dimensions of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization.
    pytorch_compatible (bool): If True perform the group normalization in
        the same order/grouping as PyTorch.
                
## mlx.nn.layers.normalization.InstanceNorm
                Applies instance normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively. Both are of size :attr:`dims`,
if :attr:`affine` is ``True``.

Args:
    dims (int): The number of features of the input.
    eps (float): A value added to the denominator for numerical stability. Default: ``1e-5``.
    affine (bool): Default: ``False``.

Shape:
  - Input: :math:`(..., C)` where :math:`C` is equal to :attr:`dims`.
  - Output: Same shape as the input.

Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> x = mx.random.normal((8, 4, 4, 16))
    >>> inorm = nn.InstanceNorm(dims=16)
    >>> output = inorm(x)

References:
    [1]: https://arxiv.org/abs/1607.08022
                
## mlx.nn.layers.normalization.LayerNorm
                Applies layer normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

[1]: https://arxiv.org/abs/1607.06450

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization
                
## mlx.nn.layers.normalization.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.normalization.RMSNorm
                Applies Root Mean Square normalization [1] to the inputs.

Computes

..  math::

    y = \frac{x}{\sqrt{E[x^2] + \epsilon}} \gamma

where :math:`\gamma` is a learned per feature dimension parameter initialized at
1.

[1]: https://arxiv.org/abs/1910.07467

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
                
## mlx.nn.layers.positional_encoding.ALiBi
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.positional_encoding.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.positional_encoding.RoPE
                Implements the rotary positional encoding.

The traditional implementation rotates consecutive pairs of elements in the
feature dimension while the default implementation rotates pairs with
stride half the feature dimensions for efficiency.

For more details see `RoFormer: Enhanced Transformer with Rotary Position
Embedding <https://arxiv.org/abs/2104.09864>`_.

Args:
    dims (int): The feature dimensions to be rotated. If the input feature
        is larger than dims then the rest is left unchanged.
    traditional (bool, optional): If set to True choose the traditional
        implementation which is slightly less efficient. Default: ``False``.
    base (float, optional): The base used to compute angular frequency for
        each dimension in the positional encodings. Default: ``10000``.
    scale (float, optional): The scale used to scale the positions. Default: ``1.0``.

Attributes:
    _cos_sin_theta_key (tuple): Cached key for the precomputed cosine and sine values.
    _cos_sin_theta_value (tuple): Cached cosine and sine values.
                
## mlx.nn.layers.positional_encoding.SinusoidalPositionalEncoding
                Implements sinusoidal positional encoding.

For more details see the paper `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

Args:
    dims (int): The dimensionality of the resulting positional embeddings.
    min_freq (float, optional): The minimum frequency expected. Default:
        ``0.0001``.
    max_freq (float, optional): The maximum frequency expected. Default:
        ``1``.
    scale (float, optional): A multiplicative scale for the embeddings.
        Default: ``sqrt(dims//2)``.
    cos_first (bool, optional): If ``True`` embed using ``[cos(x); sin(x)]``
        instead of the reverse. Default: ``False``.
    full_turns (bool, optional): If ``True`` multiply the frequencies with
        :math:`2\pi`. Default: ``False``.
                
## mlx.nn.layers.prelu
                Applies the element-wise parametric ReLU.

.. math::
    \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

where :math:`a` is an array.
                
## mlx.nn.layers.quantized.Linear
                Applies an affine transformation to the input.

Concretely:

.. math::

    y = x W^\top + b

where:
where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.quantized.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.quantized.QuantizedLinear
                Applies an affine transformation to the input using a quantized weight matrix.

It is the quantized equivalent of :class:`mlx.nn.Linear`. For now its
parameters are frozen and will not be included in any gradient computation
but this will probably change in the future.

QuantizedLinear also provides two useful classmethods to convert linear
layers to QuantizedLinear layers.

- :meth:`from_linear` returns a QuantizedLinear layer that applies the same
  linear transformation up to the quantization error.
- :meth:`quantize_module` swaps all the linear layers of the passed module
  with QuantizedLinear ones.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will not use
        a bias. (default: True).
    group_size (int, optional): The group size to use for the quantized
        weight. See :func:`~mlx.core.quantize`. (default: 64)
    bits (int, optional): The bit width to use for the quantized weight.
        See :func:`~mlx.core.quantize`. (default: 4)
                
## mlx.nn.layers.quantized.tree_flatten
                Flattens a python tree to a list of key, value tuples.

The keys are using the dot notation to define trees of arbitrary depth and
complexity.

.. code-block:: python

    from mlx.utils import tree_flatten

    print(tree_flatten([[[0]]]))
    # [("0.0.0", 0)]

    print(tree_flatten([[[0]]], ".hello"))
    # [("hello.0.0.0", 0)]

.. note::
   Dictionaries should have keys that are valid python identifiers.

Args:
    tree (Any): The python tree to be flattened.
    prefix (str): A prefix to use for the keys. The first character is
        always discarded.
    is_leaf (Callable): An optional callable that returns True if the
        passed object is considered a leaf or False otherwise.

Returns:
    List[Tuple[str, Any]]: The flat representation of the python tree.
                
## mlx.nn.layers.quantized.tree_map
                Applies ``fn`` to the leaves of the python tree ``tree`` and
returns a new collection with the results.

If ``rest`` is provided, every item is assumed to be a superset of ``tree``
and the corresponding leaves are provided as extra positional arguments to
``fn``. In that respect, :meth:`tree_map` is closer to :func:`itertools.starmap`
than to :func:`map`.

The keyword argument ``is_leaf`` decides what constitutes a leaf from
``tree`` similar to :func:`tree_flatten`.

.. code-block:: python

    import mlx.nn as nn
    from mlx.utils import tree_map

    model = nn.Linear(10, 10)
    print(model.parameters().keys())
    # dict_keys(['weight', 'bias'])

    # square the parameters
    model.update(tree_map(lambda x: x*x, model.parameters()))

Args:
    fn (Callable): The function that processes the leaves of the tree
    tree (Any): The main python tree that will be iterated upon
    rest (Tuple[Any]): Extra trees to be iterated together with tree
    is_leaf (Optional[Callable]): An optional callable that returns True if
        the passed object is considered a leaf or False otherwise.

Returns:
    A python tree with the new values returned by ``fn``.
                
## mlx.nn.layers.relu
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.layers.relu6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.layers.selu
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.layers.silu
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.layers.softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.layers.softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.layers.softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.layers.step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.layers.tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.layers.transformer.Any
                Special type indicating an unconstrained type.

- Any is compatible with every type.
- Any assumed to have all methods.
- All values assumed to be instances of Any.

Note that all the above statements are true from the point of view of
static type checkers. At runtime, Any should not be used with instance
checks.
                
## mlx.nn.layers.transformer.Dropout
                Randomly zero a portion of the elements during training.

The remaining elements are multiplied with :math:`\frac{1}{1-p}` where
:math:`p` is the probability of zeroing an element. This is done so the
expected value of a given element will remain the same.

Args:
    p (float): The probability to zero an element
                
## mlx.nn.layers.transformer.LayerNorm
                Applies layer normalization [1] on the inputs.

Computes

.. math::

    y = \frac{x - E[x]}{\sqrt{Var[x]} + \epsilon} \gamma + \beta,

where :math:`\gamma` and :math:`\beta` are learned per feature dimension
parameters initialized at 1 and 0 respectively.

[1]: https://arxiv.org/abs/1607.06450

Args:
    dims (int): The feature dimension of the input to normalize over
    eps (float): A small additive constant for numerical stability
    affine (bool): If True learn an affine transform to apply after the
        normalization
                
## mlx.nn.layers.transformer.Linear
                Applies an affine transformation to the input.

Concretely:

.. math::

    y = x W^\top + b

where:
where :math:`W` has shape ``[output_dims, input_dims]`` and :math:`b` has shape ``[output_dims]``.

The values are initialized from the uniform distribution :math:`\mathcal{U}(-{k}, {k})`,
where :math:`k = \frac{1}{\sqrt{D_i}}` and :math:`D_i` is equal to ``input_dims``.

Args:
    input_dims (int): The dimensionality of the input features
    output_dims (int): The dimensionality of the output features
    bias (bool, optional): If set to ``False`` then the layer will
      not use a bias. Default is ``True``.
                
## mlx.nn.layers.transformer.Module
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.transformer.MultiHeadAttention
                Implements the scaled dot product attention with multiple heads.

Given inputs for queries, keys and values the ``MultiHeadAttention``
produces new values by aggregating information from the input values
according to the similarities of the input queries and keys.

All inputs as well as the output are linearly projected without biases by
default.

``MultiHeadAttention`` also takes an optional additive attention mask that
should be broadcastable with ``(batch, num_heads, # queries, # keys)``. The
mask should have ``-inf`` or very large negative numbers at the positions
that should *not* be attended to.

Args:
    dims (int): The model dimensions. This is also the default
        value for the queries, keys, values, and the output.
    num_heads (int): The number of attention heads to use.
    query_input_dims (int, optional): The input dimensions of the queries.
        Default: ``dims``.
    key_input_dims (int, optional): The input dimensions of the keys.
        Default: ``dims``.
    value_input_dims (int, optional): The input dimensions of the values.
        Default: ``key_input_dims``.
    value_dims (int, optional): The dimensions of the values after the
        projection. Default: ``dims``.
    value_output_dims (int, optional): The dimensions the new values will
        be projected to. Default: ``dims``.
    bias (bool, optional): Whether or not to use a bias in the projections.
        Default: ``False``.
                
## mlx.nn.layers.transformer.Transformer
                Implements a standard Transformer model.

The implementation is based on `Attention Is All You Need
<https://arxiv.org/abs/1706.03762>`_.

The Transformer model contains an encoder and a decoder. The encoder
processes the input sequence and the decoder generates the output sequence.
The interaction between encoder and decoder happens through the attention
mechanism.

Args:
    dims (int, optional): The number of expected features in the
        encoder/decoder inputs. Default: ``512``.
    num_heads (int, optional): The number of attention heads. Default:
        ``8``.
    num_encoder_layers (int, optional): The number of encoder layers in the
        Transformer encoder. Default: ``6``.
    num_decoder_layers (int, optional): The number of decoder layers in the
        Transformer decoder. Default: ``6``.
    mlp_dims (int, optional): The hidden dimension of the MLP block in each
        Transformer layer. Defaults to ``4*dims`` if not provided. Default:
        ``None``.
    dropout (float, optional): The dropout value for the Transformer
        encoder and decoder. Dropout is used after each attention layer and
        the activation in the MLP layer. Default: ``0.0``.
    activation (function, optional): the activation function for the MLP
        hidden layer. Default: :func:`mlx.nn.relu`.
    custom_encoder (nn.Module, optional): A custom encoder to replace the
        standard Transformer encoder. Default: ``None``.
    custom_decoder (nn.Module, optional): A custom decoder to replace the
        standard Transformer decoder. Default: ``None``.
    norm_first (bool, optional): if ``True``, encoder and decoder layers
        will perform layer normalization before attention and MLP
        operations, otherwise after. Default: ``False``.
                
## mlx.nn.layers.transformer.TransformerDecoder
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.transformer.TransformerDecoderLayer
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.transformer.TransformerEncoder
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.transformer.TransformerEncoderLayer
                Base class for building neural networks with MLX.

All the layers provided in :mod:`mlx.nn.layers` subclass this class and
your models should do the same.

A ``Module`` can contain other ``Module`` instances or :class:`mlx.core.array`
instances in arbitrary nesting of python lists or dicts. The ``Module``
then allows recursively extracting all the :class:`mlx.core.array` instances
using :meth:`mlx.nn.Module.parameters`.

In addition, the ``Module`` has the concept of trainable and non trainable
parameters (called "frozen"). When using :func:`mlx.nn.value_and_grad`
the gradients are returned only with respect to the trainable parameters.
All arrays in a module are trainable unless they are added in the "frozen"
set by calling :meth:`freeze`.

.. code-block:: python

    import mlx.core as mx
    import mlx.nn as nn

    class MyMLP(nn.Module):
        def __init__(self, in_dims: int, out_dims: int, hidden_dims: int = 16):
            super().__init__()

            self.in_proj = nn.Linear(in_dims, hidden_dims)
            self.out_proj = nn.Linear(hidden_dims, out_dims)

        def __call__(self, x):
            x = self.in_proj(x)
            x = mx.maximum(x, 0)
            return self.out_proj(x)

    model = MyMLP(2, 1)

    # All the model parameters are created but since MLX is lazy by
    # default, they are not evaluated yet. Calling `mx.eval` actually
    # allocates memory and initializes the parameters.
    mx.eval(model.parameters())

    # Setting a parameter to a new value is as simply as accessing that
    # parameter and assigning a new array to it.
    model.in_proj.weight = model.in_proj.weight * 2
    mx.eval(model.parameters())
                
## mlx.nn.layers.transformer.relu
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.leaky_relu
                Applies the Leaky Rectified Linear Unit.

Simply ``mx.maximum(negative_slope * x, x)``.
                
## mlx.nn.log_sigmoid
                Applies the Log Sigmoid function.

Applies :math:`\log(\sigma(x)) = -\log(1 + e^{-x})` element wise.
                
## mlx.nn.log_softmax
                Applies the Log Softmax function.

Applies :math:`x + \log \sum_i e^{x_i}` element wise.
                
## mlx.nn.losses.binary_cross_entropy
                Computes the binary cross entropy loss.

Args:
    logits (array): The unnormalized (pre-sigmoid) predicted logits.
    targets (array): The binary target values in {0, 1}.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed binary cross entropy loss.
Examples:
    >>> import mlx.core as mx
    >>> import mlx.nn as nn
    >>> inputs = mx.array([0.105361, 0.223144, 1.20397, 0.916291])
    >>> targets = mx.array([0, 0, 1, 1])
    >>> loss = nn.losses.binary_cross_entropy(inputs, targets, "mean")
    >>> loss
    array([0.612192], dtype=float32)
                
## mlx.nn.losses.cosine_similarity_loss
                Computes the cosine similarity between the two inputs.

The cosine similarity loss is given by

.. math::

    \frac{x_1 \cdot x_2}{\max(\|x_1\|  \cdot \|x_2\|, \epsilon)}

Args:
    x1 (mx.array): The first set of inputs.
    x2 (mx.array): The second set of inputs.
    axis (int, optional): The embedding axis. Default: ``1``.
    eps (float, optional): The minimum value of the denominator used for
      numerical stability. Default: ``1e-8``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    mx.array: The computed cosine similarity loss.
                
## mlx.nn.losses.cross_entropy
                Computes the cross entropy loss.

Args:
    logits (array): The unnormalized predicted logits.
    targets (array): The target values, as class indices.
    weights (array, optional): Weights for each target. Default: ``None``.
    axis (int, optional): The axis over which to compute softmax. Default: ``-1``.
    label_smoothing (float, optional): Label smoothing factor. Default: ``0``.
    reduction (str, optional): Specifies the reduction to apply to the output:
        ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed cross entropy loss.
                
## mlx.nn.losses.gaussian_nll_loss
                Computes the negative log likelihood loss for a Gaussian distribution.

The loss is given by:

.. math::
    \frac{1}{2}\left(\log\left(\max\left(\text{vars},
    \ \epsilon\right)\right) + \frac{\left(\text{inputs} - \text{targets} \right)^2}
    {\max\left(\text{vars}, \ \epsilon \right)}\right) + \text{const.}

where ``inputs`` are the predicted means and ``vars`` are the the
predicted variances.

Args:
    inputs (array): The predicted expectation of the Gaussian distribution.
    targets (array): The target values (samples from the Gaussian distribution).
    vars (array): The predicted variance of the Gaussian distribution.
    full (bool, optional): Whether to include the constant term in the loss calculation.
        Default: ``False``.
    eps (float, optional): Small positive constant for numerical stability.
        Default: ``1e-6``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The Gaussian NLL loss.
                
## mlx.nn.losses.hinge_loss
                Computes the hinge loss between inputs and targets.

.. math::

   \text{hinge}(y, y_{\text{pred}}) = \max(0, 1 - y \cdot y_{\text{pred}})


Args:
    inputs (array): The predicted values.
    targets (array): The target values. They should be -1 or 1.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed hinge loss.
                
## mlx.nn.losses.huber_loss
                Computes the Huber loss between inputs and targets.

.. math::

    l_{\delta}(a) =
    \left\{ \begin{array}{ll}
        \frac{1}{2} a^2 & \text{for } |a| \leq \delta, \\
        \delta \left( |a| - \frac{1}{2} \delta \right) & \text{otherwise.}
    \end{array} \right.

Args:
    inputs (array): The predicted values.
    targets (array): The target values.
    delta (float, optional): The threshold at which to change between L1 and L2 loss.
      Default: ``1.0``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed Huber loss.
                
## mlx.nn.losses.kl_div_loss
                Computes the Kullback-Leibler divergence loss.

Computes the following when ``reduction == 'none'``:

.. code-block:: python

    mx.exp(targets) * (targets - inputs).sum(axis)

Args:
    inputs (array): Log probabilities for the predicted distribution.
    targets (array): Log probabilities for the target distribution.
    axis (int, optional): The distribution axis. Default: ``-1``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed Kullback-Leibler divergence loss.
                
## mlx.nn.losses.l1_loss
                Computes the L1 loss.

Args:
    predictions (array): The predicted values.
    targets (array): The target values.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

Returns:
    array: The computed L1 loss.
                
## mlx.nn.losses.log_cosh_loss
                Computes the log cosh loss between inputs and targets.

Logcosh acts like L2 loss for small errors, ensuring stable gradients,
and like the L1 loss for large errors, reducing sensitivity to outliers. This
dual behavior offers a balanced, robust approach for regression tasks.

.. math::

   \text{logcosh}(y_{\text{true}}, y_{\text{pred}}) =
        \frac{1}{n} \sum_{i=1}^{n}
        \log(\cosh(y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)}))


Args:
    inputs (array): The predicted values.
    targets (array): The target values.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed log cosh loss.
                
## mlx.nn.losses.mse_loss
                Computes the mean squared error loss.

Args:
    predictions (array): The predicted values.
    targets (array): The target values.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

Returns:
    array: The computed mean squared error loss.
                
## mlx.nn.losses.nll_loss
                Computes the negative log likelihood loss.

Args:
    inputs (array): The predicted distribution in log space.
    targets (array): The target values.
    axis (int, optional): The distribution axis. Default: ``-1``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: The computed NLL loss.
                
## mlx.nn.losses.smooth_l1_loss
                Computes the smooth L1 loss.

The smooth L1 loss is a variant of the L1 loss which replaces the absolute
difference with a squared difference when the absolute difference is less
than ``beta``.

The formula for the smooth L1 Loss is:

.. math::

  l =
      \begin{cases}
        0.5 (x - y)^2, & \text{ if } & (x - y) < \beta \\
        |x - y| - 0.5 \beta, &  & \text{otherwise}
      \end{cases}

Args:
    predictions (array): Predicted values.
    targets (array): Ground truth values.
    beta (float, optional): The threshold after which the loss changes
      from the squared to the absolute difference. Default: ``1.0``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``.

Returns:
    array: The computed smooth L1 loss.
                
## mlx.nn.losses.triplet_loss
                Computes the triplet loss for a set of anchor, positive, and negative samples.
Margin is represented with alpha in the math section.

.. math::

   \max\left(\|A - P\|_p - \|A - N\|_p + \alpha, 0\right)

Args:
    anchors (array): The anchor samples.
    positives (array): The positive samples.
    negatives (array): The negative samples.
    axis (int, optional): The distribution axis. Default: ``-1``.
    p (int, optional): The norm degree for pairwise distance. Default: ``2``.
    margin (float, optional): Margin for the triplet loss. Defaults to ``1.0``.
    eps (float, optional): Small positive constant to prevent numerical instability. Defaults to ``1e-6``.
    reduction (str, optional): Specifies the reduction to apply to the output:
      ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'none'``.

Returns:
    array: Computed triplet loss. If reduction is "none", returns a tensor of the same shape as input;
              if reduction is "mean" or "sum", returns a scalar tensor.
                
## mlx.nn.mish
                Applies the Mish function, element-wise.
Mish: A Self Regularized Non-Monotonic Neural Activation Function.

Reference: https://arxiv.org/abs/1908.08681

.. math::
    \text{Mish}(x) = x * \text{Tanh}(\text{Softplus}(x))
                
## mlx.nn.prelu
                Applies the element-wise parametric ReLU.

.. math::
    \text{PReLU}(x) = \max(0,x) + a * \min(0,x)

where :math:`a` is an array.
                
## mlx.nn.relu
                Applies the Rectified Linear Unit.

Simply ``mx.maximum(x, 0)``.
                
## mlx.nn.relu6
                Applies the Rectified Linear Unit 6.

Applies :math:`\min(\max(x, 0), 6)` element wise.
                
## mlx.nn.selu
                Applies the Scaled Exponential Linear Unit.

.. math::
    \text{selu}(x) = \begin{cases}
    \lambda x & \text{if } x > 0 \\
    \lambda \alpha (\exp(x) - 1) & \text{if } x \leq 0
    \end{cases}

where :math:`\lambda = 1.0507` and :math:`\alpha = 1.67326`.

See also :func:`elu`.
                
## mlx.nn.silu
                Applies the Sigmoid Linear Unit. Also known as Swish.

Applies :math:`x \sigma(x)` element wise, where :math:`\sigma(\cdot)` is
the logistic sigmoid.
                
## mlx.nn.softmax
                Applies the Softmax function.

Applies :math:`\frac{e^{x_i}}{\sum_j e^{x_j}}` element wise.
                
## mlx.nn.softplus
                Applies the Softplus function.

Applies :math:`\log(1 + \exp(x))` element wise.
                
## mlx.nn.softsign
                Applies the Softsign function.

Applies :math:`\frac{x}{1 + |x|}` element wise.
                
## mlx.nn.step
                Applies the Step Activation Function.

This function implements a binary step activation, where the output is set
to 1 if the input is greater than a specified threshold, and 0 otherwise.

.. math::
    \text{step}(x) = \begin{cases}
    0 & \text{if } x < \text{threshold} \\
    1 & \text{if } x \geq \text{threshold}
    \end{cases}

Args:
    threshold: The value to threshold at.
                
## mlx.nn.tanh
                Applies the hyperbolic tangent function.

Simply ``mx.tanh(x)``.
                
## mlx.nn.utils.value_and_grad
                Transform the passed function ``fn`` to a function that computes the
gradients of ``fn`` wrt the model's trainable parameters and also its
value.

Args:
    model (mlx.nn.Module): The model whose trainable parameters to compute
                           gradients for
    fn (Callable): The scalar function to compute gradients for

Returns:
    A callable that returns the value of ``fn`` and the gradients wrt the
    trainable parameters of ``model``
                
## mlx.nn.value_and_grad
                Transform the passed function ``fn`` to a function that computes the
gradients of ``fn`` wrt the model's trainable parameters and also its
value.

Args:
    model (mlx.nn.Module): The model whose trainable parameters to compute
                           gradients for
    fn (Callable): The scalar function to compute gradients for

Returns:
    A callable that returns the value of ``fn`` and the gradients wrt the
    trainable parameters of ``model``
                
## mlx.optimizers.AdaDelta
                Implementation of the AdaDelta optimizer with learning rate[1].

Our AdaDelta implementation follows the original paper. In detail,

[1]: Zeiler, M.D., 2012. ADADELTA: an adaptive learning rate method. arXiv preprint arXiv:1212.5701.

.. math::

    v_{t+1} &= \rho v_t + (1 - \rho) g_t^2 \\
    \Delta w_{t+1} &= \frac{\sqrt{u_t + \epsilon}}{\sqrt{v_{t+1} + \epsilon}} g_t \\
    u_{t+1} &= \rho u_t + (1 - \rho) \Delta w_{t+1}^2 \\
    w_{t+1} &= w_t - \lambda \Delta w_{t+1}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    rho (float, optional): The coefficient :math:`\rho` used for computing a
        running average of squared gradients. Default: ``0.9``
    eps (float, optional): The term :math:`\epsilon` added to the denominator to improve
      numerical stability. Default: `1e-8`
                
## mlx.optimizers.Adagrad
                Implementation of the Adagrad optimizer [1].

Our Adagrad implementation follows the original paper. In detail,

[1]: Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods
for online learning and stochastic optimization. JMLR 2011.

.. math::

    v_{t+1} &= v_t + g_t^2 \\
    w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    eps (float, optional): The term :math:`\epsilon` added to the
      denominator to improve numerical stability. Default: ``1e-8``
                
## mlx.optimizers.Adam
                Implementation of the Adam optimizer [1].

Our Adam implementation follows the original paper and omits the bias
correction in the first and second moment estimates. In detail,

[1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
optimization. ICLR 2015.

.. math::

    m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
    v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
    w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    betas (Tuple[float, float], optional): The coefficients
      :math:`(\beta_1, \beta_2)` used for computing running averages of the
      gradient and its square. Default: ``(0.9, 0.999)``
    eps (float, optional): The term :math:`\epsilon` added to the
      denominator to improve numerical stability. Default: ``1e-8``
                
## mlx.optimizers.AdamW
                Implementation of the AdamW optimizer [1].

Following the above convention, in contrast with [1], we do not use bias
correction in the first and second moments for AdamW. We update the weights
with a weight_decay (:math:`\lambda`) value:

[1]: Loshchilov, I. and Hutter, F., 2019. Decoupled weight decay
regularization. ICLR 2019.

.. math::

    m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
    v_{t+1} &= \beta_2 v_t + (1 - \beta_2) g_t^2 \\
    w_{t+1} &= w_t - \alpha (\frac{m_{t+1}}{\sqrt{v_{t+1} + \epsilon}} + \lambda w_t)

Args:
    learning_rate (float): The learning rate :math:`\alpha`.
    betas (Tuple[float, float], optional): The coefficients
      :math:`(\beta_1, \beta_2)` used for computing running averages of the
      gradient and its square. Default: ``(0.9, 0.999)``
    eps (float, optional): The term :math:`\epsilon` added to the
      denominator to improve numerical stability. Default: ``1e-8``
    weight_decay (float, optional): The weight decay :math:`\lambda`.
      Default: ``0``.
                
## mlx.optimizers.Adamax
                Implementation of the Adamax optimizer. It is a variant of Adam based
on the infinity norm [1].

Our Adam implementation follows the original paper and omits the bias
correction in the first and second moment estimates. In detail,

[1]: Kingma, D.P. and Ba, J., 2015. Adam: A method for stochastic
optimization. ICLR 2015.

.. math::

    m_{t+1} &= \beta_1 m_t + (1 - \beta_1) g_t \\
    v_{t+1} &= \max(\beta_2 v_t, |g_t|) \\
    w_{t+1} &= w_t - \lambda \frac{m_{t+1}}{v_{t+1} + \epsilon}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    betas (Tuple[float, float], optional): The coefficients
      :math:`(\beta_1, \beta_2)` used for computing running averages of the
      gradient and its square. Default: ``(0.9, 0.999)``
    eps (float, optional): The term :math:`\epsilon` added to the
      denominator to improve numerical stability. Default: ``1e-8``
                
## mlx.optimizers.Lion
                Implementation of the Lion optimizer [1].

Since updates are computed through the sign operation, they tend to
have larger norm than for other optimizers such as SGD and Adam.
We recommend a learning rate that is 3-10x smaller than AdamW and a
weight decay 3-10x larger than AdamW to maintain the strength
(lr * wd). Our Lion implementation follows the original paper. In
detail,

[1]: Chen, X. Symbolic Discovery of Optimization Algorithms. arXiv
preprint arXiv:2302.06675.

.. math::

    c_{t + 1} &= \beta_1 m_t + (1 - \beta_1) g_t
    m_{t + 1} &= \beta_2 m_t + (1 - \beta_2) g_t
    w_{t + 1} &= w_t - \eta (\text{sign}(c_t) + \lambda w_t)

Args:
    learning_rate (float): The learning rate :math:`\eta`.
    betas (Tuple[float, float], optional): The coefficients
      :math:`(\beta_1, \beta_2)` used for computing the gradient
      momentum and update direction. Default: ``(0.9, 0.99)``
    weight_decay (float, optional): The weight decay :math:`\lambda`. Default: ``0.0``
                
## mlx.optimizers.Optimizer
                The base class for all optimizers. It allows us to implement an
optimizer on a per-parameter basis and apply it to a parameter tree.

Attributes:
    state (OptimizerState): It holds the optimizer's state dictionary.
                
## mlx.optimizers.OptimizerState
                The optimizer state implements a recursively defined
:class:`collections.defaultdict`, namely a missing key in an optimizer
state is an :class:`OptimizerState`.

.. note::
   :meth:`OptimizerState.get` in contrast to a normal dictionary also sets
   the key to the ``default`` value if the ``key`` was not present in the
   dictionary.
                
## mlx.optimizers.RMSprop
                Implementation of the RMSprop optimizer [1].

[1]: Tieleman, T. and Hinton, G. 2012. Lecture 6.5-rmsprop, coursera: Neural networks for machine learning

.. math::

    v_{t+1} &= \alpha v_t + (1 - \alpha) g_t^2 \\
    w_{t+1} &= w_t - \lambda \frac{g_t}{\sqrt{v_{t+1}} + \epsilon}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    alpha (float, optional): The smoothing constant :math:`\alpha`.
      Default: ``0.99``
    eps (float, optional): The term :math:`\epsilon` added to the denominator
      to improve numerical stability. Default: ``1e-8``
                
## mlx.optimizers.SGD
                Stochastic gradient descent optimizer.

Updates a parameter :math:`w` with a gradient :math:`g` as follows

.. math::

    v_{t+1} &= \mu v_t + (1 - \tau) g_t \\
    w_{t+1} &= w_t - \lambda v_{t+1}

Args:
    learning_rate (float): The learning rate :math:`\lambda`.
    momentum (float, optional): The momentum strength :math:`\mu`. Default: ``0``
    weight_decay (float, optional): The weight decay (L2 penalty). Default: ``0``
    dampening (float, optional): Dampening for momentum :math:`\tau`. Default: ``0``
    nesterov (bool, optional): Enables Nesterov momentum. Default: ``False``
                
## mlx.optimizers.tree_map
                Applies ``fn`` to the leaves of the python tree ``tree`` and
returns a new collection with the results.

If ``rest`` is provided, every item is assumed to be a superset of ``tree``
and the corresponding leaves are provided as extra positional arguments to
``fn``. In that respect, :meth:`tree_map` is closer to :func:`itertools.starmap`
than to :func:`map`.

The keyword argument ``is_leaf`` decides what constitutes a leaf from
``tree`` similar to :func:`tree_flatten`.

.. code-block:: python

    import mlx.nn as nn
    from mlx.utils import tree_map

    model = nn.Linear(10, 10)
    print(model.parameters().keys())
    # dict_keys(['weight', 'bias'])

    # square the parameters
    model.update(tree_map(lambda x: x*x, model.parameters()))

Args:
    fn (Callable): The function that processes the leaves of the tree
    tree (Any): The main python tree that will be iterated upon
    rest (Tuple[Any]): Extra trees to be iterated together with tree
    is_leaf (Optional[Callable]): An optional callable that returns True if
        the passed object is considered a leaf or False otherwise.

Returns:
    A python tree with the new values returned by ``fn``.
                
## mlx.utils.defaultdict
                defaultdict(default_factory=None, /, [...]) --> dict with default factory

The default factory is called without arguments to produce
a new value when a key is not present, in __getitem__ only.
A defaultdict compares equal to a dict with the same items.
All remaining arguments are treated the same as if they were
passed to the dict constructor, including keyword arguments.
                
## mlx.utils.tree_flatten
                Flattens a python tree to a list of key, value tuples.

The keys are using the dot notation to define trees of arbitrary depth and
complexity.

.. code-block:: python

    from mlx.utils import tree_flatten

    print(tree_flatten([[[0]]]))
    # [("0.0.0", 0)]

    print(tree_flatten([[[0]]], ".hello"))
    # [("hello.0.0.0", 0)]

.. note::
   Dictionaries should have keys that are valid python identifiers.

Args:
    tree (Any): The python tree to be flattened.
    prefix (str): A prefix to use for the keys. The first character is
        always discarded.
    is_leaf (Callable): An optional callable that returns True if the
        passed object is considered a leaf or False otherwise.

Returns:
    List[Tuple[str, Any]]: The flat representation of the python tree.
                
## mlx.utils.tree_map
                Applies ``fn`` to the leaves of the python tree ``tree`` and
returns a new collection with the results.

If ``rest`` is provided, every item is assumed to be a superset of ``tree``
and the corresponding leaves are provided as extra positional arguments to
``fn``. In that respect, :meth:`tree_map` is closer to :func:`itertools.starmap`
than to :func:`map`.

The keyword argument ``is_leaf`` decides what constitutes a leaf from
``tree`` similar to :func:`tree_flatten`.

.. code-block:: python

    import mlx.nn as nn
    from mlx.utils import tree_map

    model = nn.Linear(10, 10)
    print(model.parameters().keys())
    # dict_keys(['weight', 'bias'])

    # square the parameters
    model.update(tree_map(lambda x: x*x, model.parameters()))

Args:
    fn (Callable): The function that processes the leaves of the tree
    tree (Any): The main python tree that will be iterated upon
    rest (Tuple[Any]): Extra trees to be iterated together with tree
    is_leaf (Optional[Callable]): An optional callable that returns True if
        the passed object is considered a leaf or False otherwise.

Returns:
    A python tree with the new values returned by ``fn``.
                
## mlx.utils.tree_unflatten
                Recreate a python tree from its flat representation.

.. code-block:: python

    from mlx.utils import tree_unflatten

    d = tree_unflatten([("hello.world", 42)])
    print(d)
    # {"hello": {"world": 42}}

Args:
    tree (List[Tuple[str, Any]]): The flat representation of a python tree.
                                  For instance as returned by :meth:`tree_flatten`.

Returns:
    A python tree.
                