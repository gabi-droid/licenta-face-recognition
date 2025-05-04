"""
Script pentru verificarea configurației GPU și compatibilității cu TensorFlow
"""

import tensorflow as tf
import sys
import platform
import os
import time

def check_gpu_configuration():
    """
    Verifică configurația GPU și compatibilitatea cu TensorFlow
    """
    print("=" * 50)
    print("Verificare Configurație GPU și TensorFlow")
    print("=" * 50)
    
    # Informații despre sistem
    print("\nInformații Sistem:")
    print(f"Sistem Operare: {platform.system()} {platform.release()}")
    print(f"Python Version: {sys.version}")
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Verificare GPU și DirectML
    print("\nVerificare GPU și DirectML:")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Număr GPU-uri detectate: {len(gpus)}")
        for gpu in gpus:
            print(f"GPU: {gpu}")
            device_details = tf.config.experimental.get_device_details(gpu)
            print(f"Nume GPU: {device_details}")
            
            # Verificare specifică pentru DirectML
            if 'DML' in str(device_details):
                print("\nDetalii DirectML:")
                print("GPU-ul folosește DirectML pentru accelerare")
                print("Acest lucru este normal pentru AMD GPU-uri")
    
    # Verificare performanță GPU
    print("\nVerificare Performanță GPU:")
    if gpus:
        try:
            # Test simplu de performanță cu operații matriceale
            print("Test de performanță cu operații matriceale...")
            start_time = time.time()
            
            with tf.device('/GPU:0'):
                # Creăm două matrice mari
                a = tf.random.normal([5000, 5000])
                b = tf.random.normal([5000, 5000])
                
                # Efectuăm înmulțirea
                c = tf.matmul(a, b)
                
                # Forțăm execuția
                _ = c.numpy()
            
            end_time = time.time()
            print(f"Timp de execuție: {end_time - start_time:.2f} secunde")
            print("Test de performanță GPU finalizat cu succes")
            
        except RuntimeError as e:
            print(f"Eroare la testul de performanță GPU: {e}")
    else:
        print("Nu se poate testa performanța GPU - nu există GPU disponibil")
    
    # Verificare variabile de mediu
    print("\nVerificare Variabile de Mediu:")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Nu este setat')}")
    print(f"TF_FORCE_GPU_ALLOW_GROWTH: {os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH', 'Nu este setat')}")
    print(f"TF_ENABLE_ONEDNN_OPTS: {os.environ.get('TF_ENABLE_ONEDNN_OPTS', 'Nu este setat')}")
    print(f"TF_GPU_THREAD_MODE: {os.environ.get('TF_GPU_THREAD_MODE', 'Nu este setat')}")
    
    # Verificare compatibilitate cu DirectML
    print("\nVerificare Compatibilitate DirectML:")
    try:
        # Verificăm dacă DirectML este disponibil
        from tensorflow.python.framework import test_util
        if test_util.is_gpu_available():
            print("DirectML este disponibil și funcțional")
        else:
            print("DirectML nu este disponibil")
    except:
        print("Nu s-a putut verifica disponibilitatea DirectML")
    
    # Verificare memoria GPU
    print("\nVerificare Memorie GPU:")
    if gpus:
        try:
            # Pentru DirectML, folosim o abordare diferită
            with tf.device('/GPU:0'):
                # Încercăm să alocăm o cantitate mare de memorie
                try:
                    # Încercăm să alocăm 8GB
                    test_tensor = tf.random.normal([2000, 2000, 200])
                    print("Memorie GPU suficientă pentru operații mari")
                    del test_tensor
                except:
                    print("Memorie GPU limitată pentru operații mari")
        except:
            print("Nu s-a putut verifica memoria GPU")
    
    # Verificare optimizări
    print("\nVerificare Optimizări:")
    try:
        print("JIT Compilation:", tf.config.optimizer.get_jit())
        print("Layout Optimizer:", tf.config.optimizer.get_experimental_options().get('layout_optimizer', False))
    except:
        print("Nu s-au putut verifica optimizările")
    
    print("\n" + "=" * 50)
    print("Verificare finalizată")
    print("=" * 50)

if __name__ == "__main__":
    check_gpu_configuration() 