# Hardware Library


## Getting Started on a Local Ubuntu Machine

This quick start guide will walk you through installation of Chisel and necessary dependencies:

* **[sbt:](https://www.scala-sbt.org/)** which is the preferred Scala build system and what Chisel uses.

* **[Verilator:](https://www.veripool.org/wiki/verilator)**, which compiles Verilog down to C++ for simulation. The included unit testing infrastructure uses this.


## (Ubuntu-like) Linux

Install Java.

We recommend JDK 11 or newer. If you encounter compatibility issues with older Chisel dependencies when using JDK 17, try JDK 11 instead.

```
sudo apt-get install default-jdk
```

Install `sbt`.

The repository uses `sbt` for Chisel builds. The current official installation instructions are maintained on the sbt website:

* https://www.scala-sbt.org/download/
* https://www.scala-sbt.org/1.x/docs/Installing-sbt-on-Linux.html

For Ubuntu and other Debian-based distributions, the official `apt` setup is:

```
sudo apt-get update
sudo apt-get install apt-transport-https curl gnupg -y
sudo mkdir -p /etc/apt/keyrings
curl -sL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0x2EE0EA64E40A89B84B2DF73499E82A75642AC823" | sudo gpg --dearmor -o /etc/apt/keyrings/scalasbt.gpg
echo "deb [signed-by=/etc/apt/keyrings/scalasbt.gpg] https://repo.scala-sbt.org/scalasbt/debian all main" | sudo tee /etc/apt/sources.list.d/sbt.list
sudo apt-get update
sudo apt-get install sbt
```

If the package repository is not reachable from your network environment, install the official universal package from the sbt download page and add its `bin/` directory to your `PATH`.

## Install Verilator.

We currently recommend Verilator version 4.016 or above.

We depend on _Berkeley Hardware Floating-Point Units_ for floating nodes. Therefore, before building library you need to clone hardfloat project, build it and then publish it locally on your system. Hardfloat repository has all the necessary information about how to build the project, here we only briefly mention how to build it and then publish it.

```bash
$ ./build.sh
```

`build.sh` clones the pinned `berkeley-hardfloat` revision, runs `publishLocal`, and copies the generated sources into `hardware/hardfloat/`.

If `sbt` is already installed globally, the script uses it. Otherwise it falls back to the `sbt-launch.jar` bundled in the hardfloat repository.

This hardware library references the design ideas and implementation methods from the [muIR GitHub repository](https://github.com/sfu-arch/muir-lib). We acknowledge the valuable insights provided by the muIR project, which inspired and guided the development of some hardware units in this repository.
