#!/bin/bash
set -e

cd ~/HeroDirt

echo
echo "============================================"
echo " HERO DIRT UPDATE"
echo "============================================"
echo

echo "[1/7] Updating MRMS..."
python src/get_mrms.py

echo
echo "[2/7] Updating SMAP..."
python src/get_smap.py

echo
echo "[3/7] Updating NWS..."
python src/get_nws.py

echo
echo "[4/7] Checking forcing..."
python src/check_forcing.py

echo
echo "[5/7] Running v2 soil model..."
python src/run_soil_model_v2.py

echo
echo "[6/7] Building trail map..."
python src/build_trail_map.py

echo
echo "[7/7] Publishing GitHub Pages site + source code..."

cp web/HeroDirt_trails.html web/index.html

touch web/.nojekyll

mkdir -p web/src

cp src/get_mrms.py web/src/
cp src/get_smap.py web/src/
cp src/get_nws.py web/src/
cp src/check_forcing.py web/src/
cp src/run_soil_model_v2.py web/src/
cp src/penman_drying.py web/src/
cp src/build_trail_map.py web/src/
cp src/download_osm_sangabriels.py web/src/
cp src/download_osm_sangabriels_areas.py web/src/

cp update_herodirt.sh web/update_herodirt.sh

cd web

git add index.html
git add .nojekyll
git add data/*.geojson
git add src/*.py
git add update_herodirt.sh

if git diff --cached --quiet; then

    echo "No website changes to commit."

else

    git commit -m "Update Hero Dirt forecast"
    git push

fi

cd ..

echo
echo "============================================"
echo " HERO DIRT UPDATE COMPLETE"
echo "============================================"
echo
