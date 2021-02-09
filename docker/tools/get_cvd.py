import docker
import docker.errors
import subprocess
import sys


def get_CVD(image):
    client = docker.from_env()

    current_devices = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'])
    current_devices = current_devices.decode('utf-8').strip().split('\n')
    all_devices = client.containers.run(image,
                                                ['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
                                                remove=True, runtime='nvidia')
    all_devices = all_devices.decode('utf-8').strip().split('\n')
    cuda_visible_devices = [all_devices.index(cd) for cd in current_devices]
    return ",".join([str(i) for i in cuda_visible_devices])


if __name__ == '__main__':
    cvd = get_CVD(sys.argv[1])

    print(cvd)