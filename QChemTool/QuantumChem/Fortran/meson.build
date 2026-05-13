project(
  'wrapper',
  ['fortran'],
  version : '0.1'
)

py = import('python').find_installation()

py.extension_module(
  'wrapper',
  [
    'fortranmodule.f95',
    'wrapper.f95',
  ],
  install : true,
  fortran_args : [
    '-fopenmp',
  ],
  link_args : [
    '-fopenmp',
  ],
)
