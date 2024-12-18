import rospy
import h5py
from sensor_msgs.msg import JointState  # Tipo de mensaje específico
import os
import signal
import sys

# Inicializa un diccionario para almacenar datos temporalmente
data_buffer = {
    "name": [],
    "position": [],
    "velocity": [],
    "effort": []
}

# Callback para los mensajes
def callback(msg):
    """
    Procesa los mensajes de tipo JointState y los almacena en el buffer.
    """
    data_buffer["name"].append(msg.name)
    data_buffer["position"].append(msg.position)
    data_buffer["velocity"].append(msg.velocity)
    data_buffer["effort"].append(msg.effort)

# Manejar señal de interrupción para guardar los datos
def signal_handler(sig, frame):
    rospy.loginfo("Detenido. Guardando datos en archivo HDF5...")
    save_to_hdf5()
    sys.exit(0)

# Guardar datos en HDF5
def save_to_hdf5():
    output_file = os.path.join(os.getcwd(), 'joint_states.h5')
    with h5py.File(output_file, 'w') as hdf_file:
        # Guardar cada atributo del mensaje como un dataset
        for key, values in data_buffer.items():
            if key == "name":
                # Almacena cadenas de texto
                hdf_file.create_dataset(key, data=[",".join(names) for names in values], dtype=h5py.string_dtype())
            else:
                # Almacena arrays numéricos
                hdf_file.create_dataset(key, data=values)
    rospy.loginfo(f"Datos guardados exitosamente en {output_file}")

# Función principal
def ros_to_hdf5():
    rospy.init_node('ros_jointstate_to_hdf5', anonymous=True)

    # Configura el topic al que quieres suscribirte
    topic = '/robot/left_arm/joint_states'  # Cambia esto si tu topic tiene otro nombre
    rospy.Subscriber(topic, JointState, callback)

    rospy.loginfo(f"Escuchando el topic {topic}. Presiona Ctrl+C para detener y guardar los datos.")

    # Maneja la señal de interrupción
    signal.signal(signal.SIGINT, signal_handler)
    rospy.spin()

if __name__ == '__main__':
    try:
        ros_to_hdf5()
    except rospy.ROSInterruptException:
        save_to_hdf5()