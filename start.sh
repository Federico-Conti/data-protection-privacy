#!/bin/sh

# run with 'source start.sh'

echo 'Creating python virtual environment ".venv"'
python3 -m venv .venv

echo ""
echo "Restoring backend python packages"
echo ""

./.venv/bin/python3 -m pip install -r requirements.txt

out=$?
if [ $out -ne 0 ]; then
    echo "Failed to restore backend python packages"
    exit $out
fi


echo ""
echo "Python virtual environment created successfully"
echo ""


echo "Activating the virtual environment"

echo "Activating the virtual environment"
if [ -f .venv/bin/activate ]; then
    . .venv/bin/activate

else
    echo "Cannot find the activation script"
    exit 1s
fi




#rm -rf .venv