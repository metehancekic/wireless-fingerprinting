



COMMAND="conda create -n cxnn2 python=2.7"  
eval $COMMAND

COMMAND="conda activate cxnn2"  
eval $COMMAND

COMMAND="pip install -r requirements.txt"  
eval $COMMAND

COMMAND="conda install mkl-service"  
eval $COMMAND

COMMAND="conda install -c conda-forge resampy"  
eval $COMMAND

