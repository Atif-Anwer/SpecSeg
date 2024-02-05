<a name="readme"></a>

<!-- [![Contributors][contributors-shield]][contributors-url] -->
![Python][python-shield]
![Tensorflow][tf-shield]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
<!-- [![Stargazers][stars-shield]][stars-url] -->

<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="SpecSeg_logo.png" alt="Logo" width="80" height="80">
  </a>
  <h3 align="center">SpecSeg Network for Specular Highlight Detection and Segmentation in Real-World Images</h3>

  <p align="center">
    Official repository for the article @ <a href="https://www.mdpi.com/1424-8220/22/17/6552"><strong>MDPI Sensors »</strong></a>
    <br />
  </p>
</div>


<!-- ABOUT THE PROJECT -->
## Citation

```
@Article{s22176552,
AUTHOR         = {Anwer, Atif and Ainouz, Samia and Saad, Mohamad Naufal Mohamad and Ali, Syed Saad Azhar and Meriaudeau, Fabrice},
TITLE          = {SpecSeg Network for Specular Highlight Detection and Segmentation in Real-World Images},
JOURNAL        = {Sensors},
VOLUME         = {22},
YEAR           = {2022},
NUMBER         = {17},
ARTICLE-NUMBER = {6552},
URL            = {https://www.mdpi.com/1424-8220/22/17/6552},
ISSN           = {1424-8220},
DOI            = {10.3390/s22176552}
}
```
### The <a href="#readme-top">Article PDF </a> is open access and can be downloaded directly from MDPI.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---
# Update (01/2024):
There is now a Live Demo of the SpecSeg network on Huggingface. Please feel free to test at [Huggingface SpecSeg space](https://huggingface.co/spaces/atifanwerPK/SpecSeg). 
(Note: Doesnt look correct on Safari browser due to Huggingface rendering, use Firefox/Chromimum based browsers please).


<!-- GETTING STARTED -->
# Introduction

This repository is the implementation of our paper 'SpecSeg Network for Specular Highlight Detection and Segmentation in Real-World Images'. The developed network and pretrained weights can be used for network training and testing. Please cite the paper if you use them and find them useful.

<br />
<div align="center">
    <img src="https://www.mdpi.com/sensors/sensors-22-06552/article_deploy/html/images/sensors-22-06552-g006.png" alt="Results" width="750" height="250">
  </a>
  <p align="center">
    Full article @ <a href="https://www.mdpi.com/1424-8220/22/17/6552"><strong>MDPI Sensors</strong></a>
    <br />
  </p>
</div>

## Requirements
- Tested with Python 3.7, Tensorflow 2.5+ on local GPU (CUDA 11.3)
- Tested with Python 3.7, Tensorflow 2.8+ on Google Colab
- Uses the <a href="https://github.com/qubvel/segmentation_models">Segmentation Models </a> library

## Datasets
The datasets used in the paper can be found at the following Github Repos (property of the respective authors):
1. <a href="https://github.com/jianweiguo/SpecularityNet-PSD">PSD Dataset</a>, Wu et al.
2. <a href="https://github.com/fu123456/SHIQ">SIHQ Dataset</a>, Fu et al.

<!-- USAGE EXAMPLES -->
## Usage
For training on your dataset, please change the path to the dataset root folder in the Jupyter notebook or python script.

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ROADMAP -->
## To Do:

- [x] Jupyter Notebook (for Colab)
- [x] Pretrained weights
- [ ] Python Script

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License
Distributed under the GNU General Public License v3.0 License, for research training and/or testing purposes. Pleas cite our paper if you use this code or network.
For more details, please see [Choose a license](https://choosealicense.com/licenses/gpl-3.0/) and the Liscence.txt file.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->
<!-- ## Contact
Atif Anwer - [@your_twitter](https://twitter.com/your_username) - email@example.com
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
<p align="right">(<a href="#readme-top">back to top</a>)</p> -->


<!-- ACKNOWLEDGMENTS -->
<!-- ## Acknowledgments

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search) -->




<!-- MARKDOWN LINKS & IMAGES -->
[python-shield]: https://img.shields.io/badge/Python-3.7-blue?style=for-the-badge&logo=appveyor
[tf-shield]: https://img.shields.io/badge/Tensorflow-2.8-orange?style=for-the-badge&logo=appveyor

[issues-shield]: https://img.shields.io/github/issues/Atif-Anwer/SpecSeg?style=for-the-badge
[issues-url]: https://github.com/Atif-Anwer/SpecSeg/issues
[license-shield]: https://img.shields.io/badge/License-CC-brightgreen?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/atifanwer/

<!-- Soruce: https://github.com/othneildrew/Best-README-Template/pull/73 -->
