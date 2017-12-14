cd QChemTool
tar xvzf ../pyscf.tar.gz
cd pyscf/lib/build
cmake ..
make

cd ../../../
export PYTHONPATH=$PWD/pyscf:$PYTHONPATH
