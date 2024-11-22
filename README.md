# DataFlowGen: An MLIR-based Framework for Efficient Dataflow Accelerator Generation


DataFlowGen is a scalable framework built on multi-level intermediate representation (MLIR) for efficient dataflow accelerator generation. DataFlowGen explicitly introduces a two-level IR to perform operations at suitable abstraction levels, capturing dataflow characteristics and multi-level hierarchy.
Leveraging these representations, we develop an automated optimizer that outlines the application kernel and performs dataflow transformations to derive a hardware-oriented control dataflow graph (H-CDFG). It enables concise representation and resource efficiency of hardware architectures.

## Setting this up
### Prerequisites
- python3
- cmake
- ninja
- clang and lld

Optionally, the following packages are required for the Python binding.

- pybind11
- numpy

### Clone DataFlowGen
```sh
$ git clone --recursive https://github.com/jiangnan7/DataFlowGen
$ cd DataFlowGen
```

### Build DataFlowGen
Run the following script to build DataFlowGen. Optionally, and `-j xx` to specify the number of parallel linking jobs.
```sh
$ ./build_and_run.sh
```

Make sure you have installed Anaconda.
```sh
$ conda create -n dataflowgen
```

Packages will be installed there.
```sh
$ python -m pip install -r requirements.txt
```

After the build, we suggest to export the following paths.
```sh
$ export PYTHONPATH=$(cd build && pwd)/python_packages/heteacc_core
```

### Build Hardware

Follow the instructions in [`hardware/README.md`](hardware/README.md) to install SBT and dependencies, enabling you to develop and run Chisel designs based on this library.

## Compiling C/C++

To translate the C/C++ kernel, run the:
```
$ ./thirdparty/Polygeist/build/bin/cgeist ./benchmark/doitgenTriple/doitgenTriple.cpp \ 
  -function=doitgenTriple -S  -memref-fullrank \
  -raise-scf-to-affine > doitgenTriple.mlir
```
## DataFlowGen-OPT

### IR Transformation 
To transform the initiation program, run the
```
$  ./build/bin/heteacc-opt  ./benchmark/if_loop_1/if_loop_1.mlir   --generate-dataflow \
 --analyze-memref-address  --optimize-dataflow  --generate-GEP --cse  --enhanced-cdfg \ 
 --hybird-branch-prediction --graph-init="top-func=if_loop_1" --debug-only="graph"
```

### Hardware Generation
This is a hardware library written in [Chisel](https://www.chisel-lang.org/). The core code is located in the hardware folder, which includes detailed hardware components and module implementations.

```
$ cd hardware
$ sbt "testOnly   heteacc.generator.if_loop_1DF_test"
```

