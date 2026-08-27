"""Ajustes de entorno de TensorFlow, se aplican antes de importar cualquier módulo."""

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")   #Reducir mensajes de tensorflow en consola
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  #Desactivar oneDNN para un entorno controlado
