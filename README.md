# PythonORBSlAM
--------------------
A Beginner's Python project that implemented certain parts of ORBSLAM

# What I did:
I started this project in 2020 winter break.

This is more or less a project where I learned things and realized that I actually can implement the minimum working part of it(because most of the methods used was already encapsulated in either OpenCV or sk-image), rather than a project where I actually implemented a new thing or a big thing. I restructure the codes for adding on features laters. 

I implemented and compared different feature extraction methods: Corners(CV2.GoodFeaturesToTrack), ORB(ORB.detect), evenCorners(divided each frame to 9 windows and perform Corners on each) and evenORB. The last two was inspired by the Q-tree algorithm, aiming to make the feature points spread more evenly in the picture. (And of course at the end the evenCorners prevail)

I used open3d for visualization so that I can get a pointcloud file output. But the visualization doesn't seem to be working well. 

# TO-DO:
Improve Visualization
Improve the algorithm(Filters, Qtree, g2o, etc.)
Visual BoW for Loop detection

# Dependency
OpenCV
sk-image
Open3d
NumPy

# Credits: This project was built upon those libraries

OpenCV:

@article{opencv_library,
    author = {Bradski, G.},
    citeulike-article-id = {2236121},
    journal = {Dr. Dobb's Journal of Software Tools},
    keywords = {bibtex-import},
    posted-at = {2008-01-15 19:21:54},
    priority = {4},
    title = {{The OpenCV Library}},
    year = {2000}
}

Open3d:

@article{Zhou2018,
    author    = {Qian-Yi Zhou and Jaesik Park and Vladlen Koltun},
    title     = {{Open3D}: {A} Modern Library for {3D} Data Processing},
    journal   = {arXiv:1801.09847},
    year      = {2018},
}

skImage:

@article{van2014scikit,
  title={scikit-image: image processing in Python},
  author={Van der Walt, Stefan and Sch{\"o}nberger, Johannes L and Nunez-Iglesias, Juan and Boulogne, Fran{\c{c}}ois and Warner, Joshua D and Yager, Neil and Gouillart, Emmanuelle and Yu, Tony},
  journal={PeerJ},
  volume={2},
  pages={e453},
  year={2014},
  publisher={PeerJ Inc.}
}



# Credits: I learned SLAM through those sources
高翔《视觉SLAM十四讲》 https://github.com/MeisonP/slambook
@Book{Gao2017SLAM, title={14 Lectures on Visual SLAM: From Theory to Practice}, publisher = {Publishing House of Electronics Industry}, year = {2017}, author = {Xiang Gao and Tao Zhang and Yi Liu and Qinrui Yan}, }

# Credits: I directly looked into his implementations and also learned things through his implementaion, I also ran my code on his demo video
Yang Yun, monocular SLAM
https://github.com/YunYang1994/openwork/tree/main/monocular_slam
https://yunyang1994.gitee.io/2020/12/19/%E7%94%A8-Python-%E6%89%8B%E6%92%B8%E4%B8%80%E4%B8%AA%E7%AE%80%E5%8D%95%E7%9A%84%E5%8D%95%E7%9B%AE-Slam-%E4%BE%8B%E5%AD%90/

Monocular ORBSLAM2:
@article{murTRO2015,
  title={{ORB-SLAM}: a Versatile and Accurate Monocular {SLAM} System},
  author={Mur-Artal, Ra\'ul, Montiel, J. M. M. and Tard\'os, Juan D.},
  journal={IEEE Transactions on Robotics},
  volume={31},
  number={5},
  pages={1147--1163},
  doi = {10.1109/TRO.2015.2463671},
  year={2015}
 }

# There are alternatives for you to learn SLAM
I heard good things about **Multiple view geometry in computer vision**
@Inbook{ref1,
editor="Ikeuchi, Katsushi",
title="Multiple View Geometry",
bookTitle="Computer Vision: A Reference Guide",
year="2014",
publisher="Springer US",
address="Boston, MA",
pages="513--513",
isbn="978-0-387-31439-6",
doi="10.1007/978-0-387-31439-6_100010",
url="https://doi.org/10.1007/978-0-387-31439-6_100010"
}


