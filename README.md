# Руководство к установке и использованию PyTorch 1.7.1 + DeepSpeed 0.4.2 + transformers на кластере HPC5

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
## 2. Создание environment
## 3. Установка PyTorch
## 4. Установка DeepSpeed

## 5. Установка transformers
## 6. Использование

### Установка Anaconda
