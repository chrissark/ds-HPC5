# Руководство к установке и использованию PyTorch 1.7.1 + DeepSpeed 0.4.3 + transformers на кластере HPC5

## 1. Установка Anaconda
  1. В режиме стандартного пользователя без прав root загрузите последнюю версию скрипта bash установки Anaconda (на момент написания последняя версия 2021.05): 
 
     ```
     curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh --output anaconda.sh
      ```
  2. Проверка целостности данных программы установки:

     ```
     sha256sum anaconda.sh
     ```
     Необходимо проверить вывод этой команды на соответствие [хэшу Anaconda с Python 3 на странице 64-битной версии Linux](https://docs.anaconda.com/anaconda/install/hashes/lin-3-64/) для соответствующей версии Anaconda. Если вывод соответствует хэшу, отображаемому в строке sha256, вы можете продолжать.
     
  3. Запустите скрипт установки и следуйте дальнейшим инструкциям на экране:
     
     ```
     bash anaconda.sh
     ```
     **ВАЖНО**: путь для установки Anaconda обязательно должен начинаться с ```/s/ls4```. В противном случае, если оставить путь по умолчанию, начинающийся с символической ссылки  ```/home```, Anaconda будет невозможно активировать в SLURM-сценариях.
     
     ```
     Output
     Anaconda3 will now be installed into this location:
     /home/kristina/anaconda3
     - Press ENTER to confirm the location
     - Press CTRL-C to abort the installation
     - Or specify a different location below

     [/home/kristina/anaconda3] >>> /s/ls4/users/kristina/anaconda3
      ```
  4. Активируйте установку:
     
     ```
     source .bashrc
     ```
  Более подробное описание установки и использования Anaconda [здесь](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-20-04-ru).
## 2. Создание conda environment с PyTorch и transformers
   1. Подключите CUDA 10.1:
      
      ```
      module load cuda/10.1
      ```
      
   2. Создайте environment с помощью файла [deepspeed_env.yml](./deepspeed_env.yml), в котором перечислены все необходимые зависимости, в т.ч. PyTorch и transformers:
      
      ```
      conda env create --name deepspeed_env --file deepspeed_env.yml
      ```
   3. Активируйте среду:

      ```
      conda activate deepspeed_env
      ```
      
## 3. Сборка DeepSpeed
   1. DeepSpeed нужно собрать отдельно в созданном environment. Для этого скопируйте репозиторий DeepSpeed:
      
      ```
      git clone --recursive https://github.com/microsoft/DeepSpeed
      ```
   2. Укажите Compute Capability для установленных на кластере карт в переменной ```TORCH_CUDA_ARCH_LIST```. Значения Compute Capability можно посмотреть [здесь](https://developer.nvidia.com/cuda-gpus#compute), для Nvidia Telsa K80 - 3.7, для Nvidia V100 - 7.0.
   
      ```
      export TORCH_CUDA_ARCH_LIST="3.7"
      ```
  3. Перейдите в директорию DeepSpeed. Выполните следующую команду:
  
     ```
     pip install .
     ```
  4. Проверьте правильность установки DeepSpeed. Для этого выполните следующую команду:
     
     ```
     ds_report
     ```  
     Вывод команды должен выглядеть так:
     ![изображение](https://user-images.githubusercontent.com/64375679/124972392-f6e32500-e032-11eb-9e85-3ff39282653d.png)

 
