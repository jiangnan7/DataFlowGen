
cd ..
mkdir hardfloat
cd hardfloat 
git clone https://github.com/ucb-bar/berkeley-hardfloat.git && cd berkeley-hardfloat
mv * ../ && rm -rf berkeley-hardfloat
git reset --hard 70455e53f233a06cb5a342d125e22b7b1505c271
sbt "publishLocal"

