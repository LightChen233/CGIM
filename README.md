# CGIM: A Cycle Guided Interactive Learning Model for Consistency Identification in Task-oriented Dialogue

<img src="img/pytorch.png" width="10%"> [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

This repository contains the PyTorch implementation and the data of the paper: **CGIM: A Cycle Guided Interactive Learning Model for Consistency Identification in Task-oriented Dialogue**. **[Libo Qin](https://scholar.google.com/citations?user=8lVpK1QAAAAJ)**, [Qiguang Chen](https://github.com/LightChen233), [Tianbao Xie](https://tianbaoxie.com/),[Qian Liu](https://siviltaram.github.io/), [Shijue Huang](https://github.com/JoeYing1019), [Wanxiang Che](http://ir.hit.edu.cn/~car/), [Yu Zhou](https://www.cs.columbia.edu/~zhouyu/).  ***COLING2022***.[[PDF]]() .



<div>
<img src="./img/SCIR_logo.png" width="70%"><img src="./img/Columbia_logo.png" width="30%">
</div>




This code has been written using PyTorch >= 1.1. If you find this code useful for your research, please consider citing the following paper:

<pre>
@misc{xxx,
      title={CGIM: A Cycle Guided Interactive Learning Model for Consistency Identification in Task-oriented Dialogue}, 
      author={Libo Qin and Qiguang Chen and Tianbao Xie and Qian Liu and Shijue Huang and Wanxiang Che and Yu Zhou},
      year={2022},
      eprint={xxx},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
</pre>


## Network Architecture

<img src="/img/model.png" alt=" " style="zoom:67%;" />


## Prerequisites

This codebase was developed and tested with the following settings:

```
-- scikit-learn==0.23.2
-- numpy==1.19.1
-- pytorch==1.1.0
-- fitlog==0.9.13
-- tqdm==4.49.0
-- sklearn==0.0
-- transformers==3.2.0
```

We highly suggest you using [Anaconda](https://www.anaconda.com/) to manage your python environment. If so, you can run the following command directly on the terminal to create the environment:

```
conda env create -f py3.6pytorch1.1_.yaml
```

## How to run it

The script **train.py** acts as a main function to the project, you can run the experiments by the following commands:

```shell
python -u train.py --cfg KBRetriver_DC_BERT_INTERACTIVE/KBRetriver_DC_BERT_INTERACTIVE.cfg
```

The parameters we use are configured in the `configure`. If you need to adjust them, you can modify them in the relevant files or append parameters to the command.

Finally, you can check the results in `logs` folder. Also,  you can run fitlog command to visualize the results:

```shell
fitlog log logs/
```

## Model Performance

<table>
	<tr>
	    <th> Model </th>
	    <th>QI F1</th>
        <th>HI F1</th>
        <th>KBI F1</th>
        <th>Overall Acc</th>
	</tr >
	<tr>
        <td>BERT (Devlin et al., 2019)</td>
	    <td>0.691</td>
        <td>0.555</td>
        <td>0.740</td>
        <td>0.500</td>
	</tr>
	<tr>
        <td>RoBERTa (Liu et al., 2019)</td>
	    <td>0.715</td>
        <td>0.472</td>
        <td>0.715</td>
        <td>0.500</td>
	</tr>
	<tr>
	    <td>XLNet (Yang et al., 2020)</td>
	    <td>0.725</td>
        <td>0.487</td>
        <td>0.736</td>
        <td>0.509</td>
	</tr>
	<tr>
	    <td>Longformer (Beltagy et al., 2020)</td>
	    <td>0.717</td>
        <td>0.500</td>
        <td>0.710</td>
        <td>0.497</td>
	</tr>
	<tr>
	    <td>BART (Lewis et al., 2020)</td>
	    <td>0.744</td>
        <td>0.510</td>
        <td>0.761</td>
        <td>0.513</td>
	</tr>
    <hline></hline>
    <tr style="background-color=#FFFFF">
	    <td>CGIM(Our)</td>
	    <td><b>0.764</b></td>
        <td><b>0.567</b></td>
        <td><b>0.772</b></td>
        <td><b>0.563</b></td>
	</tr>
</table>
