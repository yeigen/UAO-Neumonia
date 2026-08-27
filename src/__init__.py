import os 

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   #Reducir mensajes de tensorflow en consola
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  #Desactivar optimizaciones de oneDNN de TF, para un entorno mas controlado
