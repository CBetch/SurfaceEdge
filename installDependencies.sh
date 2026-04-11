packages=(torch transformers websockets requests atproto) # maverick-coref requires export PYTHONUTF8=1. 
for item in "${packages[@]}"; do
    if python -m pip show $item &> /dev/null; then
        echo "Skipping install of $item"
    else
        echo "Installing $item..."
        python -m pip install $item &> /dev/null
        if [ $? -eq 0 ]; then
            echo "Completed install of $item"
        else
            echo "Failed to install $item"
        fi 
    fi
done 
