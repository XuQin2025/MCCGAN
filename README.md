\# MCCGAN for Mg Alloys



**A multidimensional continuous conditional generative adversarial network to link composition–process–microstructure–properties in Mg alloys**



This repository provides the supporting code for our work on a multidimensional continuous conditional GAN (MCCGAN) for microstructure generation in Mg alloys. The model is designed to learn the relationship between composition + processing parameters and microstructure images, and to support downstream integration with property prediction / surrogate models in a composition–process–microstructure–properties framework.



---



\## Overview



Microstructure evolution in Mg alloys is strongly influenced by both alloy composition and processing conditions. Experimental exploration is expensive and sparse. To address this, we develop an MCCGAN that:



\* uses a continuous-valued condition vector

\* generates microstructure images conditioned on these variables,

\* supports interpolation and  in condition space within the training range,

\* supports outputs for inputs outside the range, its accuracy is not guaranteed.



---

\## Installation



\### Option 1: pip (recommended for quick setup)



```bash

pip install torch torchvision numpy pillow tqdm

```



\### Option 2: create a virtual environment (recommended for reproducibility)



```bash

python -m venv mccgan\_env

\# Windows

mccgan\_env\\Scripts\\activate

\# Linux/macOS

source mccgan\_env/bin/activate



pip install torch torchvision numpy pillow tqdm

```



---

\## Data Preparation



\### Expected Folder Structure



The training script expects a directory structure like:



```text

new2/

├── AT66\_350\_4/

│   ├── img1.tif

│   ├── img2.tif

│   └── ...

├── AT69\_430\_12/

│   ├── img1.tif

│   ├── img2.tif

│   └── ...

└── ...

```



\### Folder Naming Convention



Each subfolder name encodes one condition tuple in the format:



```text

ATxy\_TTT\_HH

```



Example:



```text

AT69\_390\_12

```

which is parsed as:



\* `A = 6`

\* `T = 9`

\* `Temp = 390`

\* `Time = 12`



> Please make sure the folder names strictly follow this convention if you use the provided parser.



---

\## Training (example)



```bash

python MCCGAN\_train.py --data\_root new2 --epochs 200 --batch\_size 16

```

---

\## Inference (example)



```bash

python MCCGAN\_infer.py --weights runs/MCCGAN/weights/G\_final.pt --out\_dir runs/MCCGAN/infer\_12x10

```



---

\## Citation



If you use this code in your research, please cite our paper:



A multidimensional continuous conditional generative adversarial network to link composition-process-microstructure-properties in Mg alloys

