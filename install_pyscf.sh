cd QChemTool
tar xvzf ../pyscf.tar.gz
cd pyscf/lib/build
cmake ..
make

cd ../../../
export PYTHONPATH=$PWD"/QChemTool":$PYTHONPATH
echo ' ' >> ~/.bashrc
echo '# pyscf path:' >> ~/.bashrc
echo 'export PYTHONPATH='$PWD'/QChemTool:$PYTHONPATH' >> ~/.bashrc
