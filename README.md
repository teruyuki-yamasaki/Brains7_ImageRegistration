# 第7回 Brain(s)コンテスト - 医用画像レジストレーション
※ The English version is below. 

2021年12月1日から2022年1月17日に行われた、
[第7回Brain(s)コンテスト](https://fujifilmdatasciencechallnge.mystrikingly.com/)に参加したので、
その振り返りをここにまとめたいと思います。

Brain(s)コンテストは、FUJIFILM AI Academy Brain(s)さん主催のデータサイエンスコンペであり、
画像やマテリアルズインフォマティクスなどのAI技術に関する問題が出題されてきました。
私が参加した第7回のテーマは医用画像レジストレーションであり、
MRIの脳画像を用いて、２枚の画像間での位置合わせを行うことが主な課題でした。

問題はQ1、Q2、Q3の3問からなり、それぞれ、
- Q1: アフィン変換行列に関する問題
- Q2: 2次元医用画像レジストレーション
- Q3: 3次元医用画像レジストレーション

であり、Q2とQ3の精度を参加者同士が競い合うというものでした。

結果としては、Q3を１位で表彰していただくことができました。
また、Q2もスコア自体は4人同率1位のスコアのうち一つを取ることができました。
ただし、4人並んだQ2については、解法の内容や外れ値の少なさなど総合的により優れていた方がいらっしゃったので、
運営の方々の判断で、１位は別の方に贈られています。

## Q1: アフィン変換行列の操作
Q1は、与えられた2次元の点群に対して、指示されたアフィン変換を行い、その正誤を確かめる問題でした。
問題の詳細は省略しますが、以下のような結果になりました。
問題文で誘導がされていた通り、各変形操作に対応するアフィン変換行列を掛け合わせることで解きやすいようになっていました。
注意点は、回転や拡大縮小の際に、操作の中心が原点でない場合に、その操作の前後で平行移動することでした。
後のQ2やQ3で使用するアフィン変換行列の操作に慣れるための目的で用意された問題だったようです。

## Q2: 2次元医用画像レジストレーション
Q2は、2次元医用画像レジストレーションであり、与えられた２枚のペアの脳画像に対して、その画像間の変形を推定する問題でした。
画像データはIXI DatasetのT1画像から得られた脳のMRI断層画像であり、
元の画像がsource画像(256x256 pixels)、それに謎の変形を加えて得られる画像がtarget画像(256x256 pixels)として与えられ、
その謎の変形を推定するというものでした。

source画像には、10個のキーポイントのピクセル座標値が与えられており、
それらのtaregt画像上における対応点のピクセル座標値を推定し、
その推定値とground truthとのユークリッド距離の平均値がスコアとして計算されるというものでした。



## Q3: 3次元医用画像レジストレーション




# 7th Brain(s) Contest - Medical Image Registration

This repository is about [7th Brain(s) Contest](https://fujifilmdatasciencechallnge.mystrikingly.com/), 
a data science compation that was held by FUJIFILM AI Academy Brain(s) 
from December 1, 2021 to January 17, 2022. 
This competition is composed of three problems, Q1, Q2 and Q3, each regarding image registration. 


The description of my solutions is presented in [this document](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/docs/brains7.pdf). 

My implementations are available as follows: 
- Q1: [Q1main.py](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/code/Q1/mainQ1.py)
- Q2: [Q2main_itk.py](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/code/Q2/mainQ2_itk.py)
- Q3: [Q3main.py](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/code/Q3/mainQ3.py)
