cd QChemTool
tar xvzf ../pyscf.tar.gz
cd pyscf/lib/build
cmake ..
make

cd ../../../
export PYTHONPATH=$PWD:$PYTHONPATH
echo ' ' >> ~/.bashrc
echo '# pyscf path:' >> ~/.bashrc
echo 'export PYTHONPATH='$PWD':$PYTHONPATH' >> ~/.bashrc
