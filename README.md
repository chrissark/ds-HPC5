# Руководство к установке и использованию PyTorch 1.7.1 + DeepSpeed 0.4.3 + transformers 4.6.1 на кластере HPC5

Для обучения моделей-трансформеров в распределенном режиме с помощью DeepSpeed на кластере требуется создать среду в Anaconda с установленными библиотеками PyTorch 1.7.1 и transormers 4.6.1, затем в этой среде собрать DeepSpeed. Ниже приведена инструкция по установке Anaconda, созданию conda environment, сборке DeepSpeed. Также в репозитории содержатся файлы примера обучения GPT-2: скрипт обучения [ds_gpt2.py](ds_gpt2.py), конфигурационный файл JSON для DeepSpeed [ds_config.json](ds_config.json) и SLURM-сценарий запуска задачи [slurm_ds_gpt2.sh](slurm_ds_gpt2.sh).

 - [Установка Anaconda](#установка-anaconda) 
 - [Создание conda environment с PyTorch и transformers](#создание-conda-environment-с-pytorch-и-transformers)
 - [Сборка DeepSpeed](#сборка-deepspeed)
 - [Пример программы для обучения GPT-2](#пример-программы-для-обучения-gpt-2)
    - [Интеграция transformers с DeepSpeed](#интеграция-transformers-с-deepspeed)
 - [Запуск параллельной задачи SLURM](#запуск-параллельной-задачи-slurm)
 - [Ссылки](#ссылки)

## Установка Anaconda
  1. В режиме стандартного пользователя без прав root загрузите последнюю версию скрипта bash установки Anaconda (на момент написания последняя версия 2021.05): 
 
     ```
     curl https://repo.anaconda.com/archive/Anaconda3-2021.05-Linux-x86_64.sh --output anaconda.sh
      ```
  2. Проверьте целостность данных программы установки с помощью следующей команды:

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
## Создание conda environment с PyTorch и transformers
   1. Подключите CUDA 10.1:
      
      ```
      module load cuda/10.1
      ```
      
   2. Создайте environment с помощью файла [deepspeed_env.yml](./deepspeed_env.yml), в котором перечислены все необходимые зависимости, в т.ч. [PyTorch-gpu 1.7.1](https://anaconda.org/conda-forge/pytorch-gpu/files)  и [transformers](https://huggingface.co/transformers/master/index.html):
      
      ```
      conda env create --name ds_env --file deepspeed_env.yml
      ```
   3. Активируйте среду:

      ```
      conda activate ds_env
      ```
   Подключать CUDA 10.1 следует при каждой активации созданной среды. Так, ```module load cuda/10.1``` нужно прописывать в SLURM-сценариях перед ```conda activate```.
      
## Сборка DeepSpeed
   1. DeepSpeed нужно собрать отдельно в созданном environment. Для этого скопируйте репозиторий DeepSpeed:
      
      ```shell
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
     ![изображение](https://user-images.githubusercontent.com/64375679/125061956-8c73c880-e0b6-11eb-927a-02b48c54ade4.png)
     
     Утилиты DeepSpeed, такие как cpu_adam, fused_adam и др., компилируются динамически (JIT) во время запуска программы. Можно предустановить некоторые из них, устанавливая различные переменные, равными 1, например:
     
     ```
     DS_BUILD_CPU_ADAM=1 DS_BUILD_FUSED_ADAM=1 pip install .
     ```
     Полный список переменных и подробности установки DeepSpeed приведены [здесь](https://www.deepspeed.ai/tutorials/advanced-install/) и [здесь](https://huggingface.co/transformers/master/main_classes/deepspeed.html). 
     
     **Замечание**: ```DS_BUILD_OPS=1 pip install .``` - предварительная установка всех утилит сразу, завершается с ошибкой при ```TORCH_CUDA_ARCH_LIST="3.7"```, но работает при ```TORCH_CUDA_ARCH_LIST="7.0"```.

## Пример программы для обучения GPT-2

Для обучения GPT-2 с помощью DeepSpeed, PyTorch и библиотеки transformers потребуется скрипт [ds_gpt2.py](ds_gpt2.py) и конфигурационный файл [ds_config.json](ds_config.json). На кластере файлы лежат в директории ```/s/ls4/users/kristina/nlp/GPT-2```. Подробнее о конфигурационных JSON файлах DeepSpeed [здесь](https://www.deepspeed.ai/docs/config-json/). Для обучения GPT-2 используется датасет [wikitext](https://huggingface.co/datasets/wikitext).

**Замечание**: при обучении моделей на кластере датасеты, модели и др. данные следует загружать **локально**, поскольку на узлах отсутствует подключение к Интернет. В данном примере загружается версия датасета, лежащая в ```/s/ls4/users/kristina/nlp/GPT-2/wikitext```, сохраненная после предыдущего запуска программы на головном узле fjord1, на котором есть Интернет, с помощью метода datasets [save_to_disk()](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset):

```python
datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
datasets.save_to_disk('./wikitext')
```
Загрузка датасета с помощью [load_from_disk()](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset):

```python
from datasets import load_from_disk
datasets = load_from_disk('./wikitext')
```

Модель и токенизатор предварительно были сохранены при запуске программы на fjord1 с помощью метода [save_pretrained()](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.save_pretrained) библиотеки transofmers:

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.save_pretrained('/s/ls4/users/kristina/nlp/GPT-2/distgpt2-tokenizer/')
model = GPT2LMHeadModel.save_pretrained('/s/ls4/users/kristina/nlp/GPT-2/distgpt2')
```
Последующая загрузка модели и токенизатора на узлах кластера происходит аналогично, но с помощью метода [from_pretrained()](https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig.from_pretrained):

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('/s/ls4/users/kristina/nlp/GPT-2/distgpt2-tokenizer/')
model = GPT2LMHeadModel.from_pretrained('/s/ls4/users/kristina/nlp/GPT-2/distgpt2')
```

### Интеграция transformers с DeepSpeed

Обучение в примере происходит с помощью специального класса [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) библиотеки transformers. Для подключения DeepSpeed объекту типа Trainer нужно передать в качестве аргумента путь к конфигурационному JSON-файлу в TrainingArguments:

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    "test-clm",
    learning_rate=2e-5,
    num_train_epochs=1,
    deepspeed = "./ds_config.json"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets["train"],
)
```

Подробнее об интеграции transformers с DeepSpeed, а также о том, как обучать модели без Trainer с помощью PyTorch и DeepSpeed [здесь](https://huggingface.co/transformers/master/main_classes/deepspeed.html).

## Запуск параллельной задачи SLURM

Пример сценария для запуска обучения GPT-2 на 5 узлах с 4 GPU: [slurm_ds_gpt2.sh](slurm_ds_gpt2.sh).
Число узлов регулируется параметром ```--nodes```. Для того, чтобы задействовать все 4 графические карты на узле, параметр ```ntasks-per-node``` должен быть равен 4.

Запуск вычисленний осуществляется с помощью следующей команды:

```
mpirun python3 ds_gpt2.py
```

## Ссылки
- [Установка Anaconda](https://www.digitalocean.com/community/tutorials/how-to-install-the-anaconda-python-distribution-on-ubuntu-20-04-ru)
- [PyTorch-gpu 1.7.1](https://anaconda.org/conda-forge/pytorch-gpu/files)
- [transformers](https://huggingface.co/transformers/master/index.html)
- [CUDA GPUs Compute Capability](https://developer.nvidia.com/cuda-gpus#compute)
- [DeepSpeed Advanced Install](https://www.deepspeed.ai/tutorials/advanced-install/)
- [DeepSpeed Integration (transformers)](https://huggingface.co/transformers/master/main_classes/deepspeed.html)
- [DeepSpeed конфигурационные файлы JSON](https://www.deepspeed.ai/docs/config-json/)
- [Wikitext dataset](https://huggingface.co/datasets/wikitext)
- [transformers loading methods](https://huggingface.co/docs/datasets/package_reference/loading_methods.html#datasets.load_dataset)
- [transformers save_pretrained()](https://huggingface.co/transformers/main_classes/model.html#transformers.PreTrainedModel.save_pretrained)
- [transformers from_pretrained()](https://huggingface.co/transformers/main_classes/configuration.html#transformers.PretrainedConfig.from_pretrained)
- [Trainer](https://huggingface.co/transformers/main_classes/trainer.html)


