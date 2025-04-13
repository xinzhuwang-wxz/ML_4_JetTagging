[[_TOC_]]
## Prepare
```shell
mkdir build install run
```
Then change the paths in env.sh:
```shell
export PATH="/your path/install/bin:$PATH"
export DATA_DIR="/your path/merge"  # where the data is
# merge structure:
  
# ├── merge
# │   ├── bb
# │   │   ├── root 1
# │   │   └── ...
# │   ├── cc
# │   │   ├── root 1
# │   │   └── ...
...
# │   ├── gg
# │   │   ├── root 1
# │   │   └── ...
# │   ├── val
# │   │   ├── all for val...
# │   └── test
# │       ├── all for test...

```
## compile
```shell
/cvmfs/container.ihep.ac.cn/bin/hep_container shell CentOS7
source env.sh
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../install 
make -j10  
make install
```

## Test
```
cd ../run
bdt_joi
```

## Note
- for saving time, we sample the data for bdt analysis, you can find in Bstrain.cpp:  (std::rand() < 0.05 * RAND_MAX) 
- the method of dealing with big data needs to be improved
- performance of bdt is not good, we can try to use the more data or more features
- current performance plots are in default_xgb


## Acknowledgement
thanks to <https://github.com/ZHUYFgit/CEPC-Jet-Origin-Identification/tree/main/fast_simulation>
          <https://code.ihep.ac.cn/zyjonah/cepc_hss_scripts>
Jet tagging paper: <https://arxiv.org/abs/2310.03440>