# 第7回 Brain(s)コンテスト - 医用画像レジストレーション
※ The English version is below. 

2021年12月1日から2022年1月17日に行われた、
[第7回Brain(s)コンテスト](https://fujifilmdatasciencechallnge.mystrikingly.com/)に参加したので、
その振り返りをここにまとめたいと思います。

詳細な説明は[こちらのPDF](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/docs/brains7.pdf)にまとめました。

Brain(s)コンテストは、FUJIFILM AI Academy Brain(s)さん主催のデータサイエンスコンペであり、
画像やマテリアルズインフォマティクスなどのAI技術に関する問題が出題されてきました。
私が参加した第7回のテーマは、画像レジストレーションであり、
MRIの脳画像を用いて、画像ペア間での位置合わせを行うことが主な課題でした。

問題はQ1、Q2、Q3の3問からなり、それぞれ、
- Q1: アフィン変換行列に関する問題
- Q2: 2次元医用画像レジストレーション
- Q3: 3次元医用画像レジストレーション

であり、Q2とQ3の精度を参加者同士が競い合うというものでした。

結果としては、Q3を１位で表彰していただくことができました。
また、Q2もスコア自体は4人同率1位のスコアのうち一つを取ることができました。
ただし、4人並んだQ2については、解法の内容や外れ値の少なさなど総合的により優れていた方がいらっしゃったので、
運営の方々の判断で１位はその方に贈られています。

## Q1: アフィン変換行列の操作
Q1は、与えられた2次元の点群に対して指示されたアフィン変換を行って変換後の座標を導出し、その正誤を確かめる問題でした。
問題の詳細は省略しますが、以下のような結果になりました。
問題文で誘導がされていた通り、各変形操作に対応するアフィン変換行列を掛け合わせることで解きやすいようになっていました。
注意点は、回転や拡大縮小の際に、操作の中心が原点でない場合に、その操作の前後で平行移動することでした。
後のQ2やQ3で使用するアフィン変換行列の操作に慣れるための目的で用意された問題だったようです。

## Q2: 2次元医用画像レジストレーション
### 問題内容
Q2は、2次元医用画像レジストレーションであり、与えられた２枚のペアの脳画像に対して、その画像間の変形を推定する問題でした。
画像データはIXI DatasetのT1画像から得られた脳のMRI断層画像であり、
元の画像がsource画像(256x256 pixels)、そしてそれに謎の変形を加えて得られる画像がtarget画像(256x256 pixels)として与えられ、
２枚の画像からその謎の変形を推定するというものでした。

source画像には、10個程度のキーポイントのピクセル座標値が与えられており、
それらのtaregt画像上における対応点のピクセル座標値を推定し、
その推定値とground truthとのユークリッド距離の平均値がスコアとして計算されるというものでした。

### アプローチ
最初は、テンプレートマッチングでどこまで精度を上げられるか試していました。
NCC (normalized cross correlation;正規化相互相関)という画像パッチ間の類似度の指標を用いて、
各キーポイントの周りのパッチとtarget画像の各ピクセル周りパッチの類似度を逐一計算し、
最大値を返すピクセルが対応点であるとしてどんな結果が出るかを確かめました。
本課題では、9x9が最適なパッチサイズだったようで、このときに2.056まで出ました。

テンプレートマッチングの結果は、肉眼では概ねよく一致しており、一致している場合はかなり信頼できたのですが、
一方でそれだけでは明らかな外れ値も多く含まれました。
そこで、キーポイント以外にも多くの点をサンプリングして同様にテンプレートマッチングの推定を行い、
それらのデータ点を用いてRANSACや最小二乗法を用いてそれらの点群間のアフィン変換を推定しました。
そして、キーポイントのテンプレートマッチングの結果が推定アフィン変換によるリプロジェクションの結果と大きく異なる場合は外れ値とみなし、
よりたしからしい後者に置き換えるという方法を試しました。その閾値などのパラメータを調整してどこまで行くか試したら、1.297まで行きました。
[この方針](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/code/Q2/mainQ2_mine.py)ではこの辺りで頭打ちとなり、
ここから変形モデルとして非線形を組み込む方法を考えました。

非線形変形でよくきくThin-plate splineやB-splineでレジストレーションしたらどうなるのか気になり、
調べていると、問題文でも紹介のあったITKの[ITKElastix](https://github.com/InsightSoftwareConsortium/ITKElastix)という
画像レジストレーション用の有名なソフトウェアElastixのPython用インターフェイスがあることを知りました。
実装方法が分からなかったので[こちらの記事](https://qiita.com/39MIFU/items/06aa11512937cae8f0a7)の内容をほとんどそのまま頼りに
本問題用に改変する形で動かしてみました。すると、うまく動き（動いてしまい）、0.856というスコアになりました。
ルール上はOKということになっていましたが、流石にこれではまずいと思い、
これにテンプレートマッチングの結果を組み合わせたり、自分で実装しようと試みましたが、なかなかこれを超えるスコアは難しく、
とりあえず置いておき、代わりにQ3に取り組むことにしました。最終的に、Q3を一通り解いた後の時間が残されておらず、
Q2はこのまま提出することになりました。個人的には、流石にこれでは終われないと思っていたので、本当にQ3で挽回できたのが救いでした。

## Q3: 3次元医用画像レジストレーション
### 問題内容
Q3は、3次元のMRI画像ボリュームデータのペアの間でのレジストレーションが課題でした。
Q2と同じく、IXI Datasetのデータが用いられていましたが、
source画像がT2、taregt画像がPD強調画像で与えれており、異なるモダリティの3次元画像のレジストレーションになっていました。
target画像には謎の変形が加わっていて、source画像にはキーポイントの3次元座標値が与えられており、
taregt画像上におけるキーポイントの対応点を推定し、その推定点とground truthのユークリッド距離の平均値でスコアが評価されました。

### アプローチ
正直最初は全然分からなかったのですが、
[scipy.ndimage.affine_transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.affine_transform.html)
のようなライブラリーを使えば、3次元ボリュームデータのアフィン変換が容易にできることを知り、
これを使って、3次元データのアフィン変換行列のパラメータを最適化することを方針にしてみました。

いくつか試そうとしましたが、実装上、兆しが見えてきたのは
ボクセル値の差の二乗和SSD(the Sum of Squared Differences)を類似度の指標として、
非線形最小二乗法のソルバー([scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html))
を使って最適化するというものでした。

詳細な説明は[PDF](https://github.com/teruyuki-yamasaki/Brains7_ImageRegistration/blob/main/docs/brains7.pdf)の方に譲りますが、概要は以下の通りとなります。

求めたい3次元空間のアフィン変換は、アフィン変換行列のパラメータ12個<img src="https://latex.codecogs.com/svg.image?\textbf{p}&space;=&space;(a_{11},&space;\cdots,&space;b_3)^T" title="\textbf{p} = (a_{11}, \cdots, b_3)^T" />で表されます:

<img src="https://latex.codecogs.com/svg.image?\begin{pmatrix}a_{11}&space;&&space;a_{12}&space;&&space;a_{13}&space;&&space;b_1&space;\\a_{21}&space;&&space;a_{22}&space;&&space;a_{23}&space;&&space;b_2&space;\\a_{31}&space;&&space;a_{32}&space;&&space;a_{33}&space;&&space;b_3&space;\\&space;&space;&space;&space;&space;0&space;&&space;&space;&space;&space;&space;&space;0&space;&&space;&space;&space;&space;&space;&space;0&space;&&space;&space;&space;1&space;\\\end{pmatrix}" title="\begin{pmatrix}a_{11} & a_{12} & a_{13} & b_1 \\a_{21} & a_{22} & a_{23} & b_2 \\a_{31} & a_{32} & a_{33} & b_3 \\ 0 & 0 & 0 & 1 \\\end{pmatrix}" />

このアフィン変換を用いて
source画像<img src="https://latex.codecogs.com/svg.image?I_{source}" title="I_{source}" />を変形させて得られる
ワープ画像<img src="https://latex.codecogs.com/svg.image?I_{warped}" title="I_{warped}" />と、
target画像<img src="https://latex.codecogs.com/svg.image?I_{target}" title="I_{target}" />の差をとって得られる画像データ

<img src="https://latex.codecogs.com/svg.image?\textbf{r}(\textbf{p})&space;:=&space;I_{warped}(\textbf{p})&space;-&space;I_{target}&space;" title="\textbf{r}(\textbf{p}) := I_{warped}(\textbf{p}) - I_{target} " />

は、パラメータ<img src="https://latex.codecogs.com/svg.image?\textbf{p}" title="\textbf{p}" />の関数であり、
この関数<img src="https://latex.codecogs.com/svg.image?\textbf{r}(\textbf{p})" title="\textbf{r}(\textbf{p})" />と
パラメータ<img src="https://latex.codecogs.com/svg.image?\textbf{p}" title="\textbf{p}" />の初期値を
非線形最小二乗法のソルバー([scipy.optimize.least_squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html))
に渡せば、うまく動けば最適なパラメータ<img src="https://latex.codecogs.com/svg.image?\textbf{p}" title="\textbf{p}" />を見つけてきてくれます。

ただし、今回はsourceとtargetで異なるタイプの強調画像(それぞれT2とPD)が使われているため、
これだけでは少し動いたものの、ほとんど動いてくれませんでした。

そこで、苦肉の策として、source(T2)とtarget(PD)両画像データの各z平面の輝度勾配を計算し、
輝度勾配のL2ノルム<img src="https://latex.codecogs.com/svg.image?\sqrt{I_x^2&space;&plus;&space;I_y^2}" title="\sqrt{I_x^2 + I_y^2}" />の2次元データを
z方向に再度積み重ねて得られるボリュームデータを作成し、そうして得られたデータを上の最適化計算に投げてみることにしました。

すると最終的にうまく動いてくれて、0.536というスコアを得ることができました。締め切りの二日前にようやく動いてくれて、本当に焦りましたがよかったです。

以上が私の解法の概要になります。まずは画像処理ベースの手法でできるところまでやってみて、
時間に余裕があればディープベースの手法も試したいと思っていましたが、そこまではさすがに時間が足りませんでした。
Q2は試行錯誤した後に、納得しづらい解になってしまったので、その分Q3をちゃんと自力で動かせた点が収穫でした。


## 最後に
本大会の運営に携わってくださったFUJIFILM AI Academyの関係者の皆様にここで改めて感謝を申し上げたいと思います。
大会を通じて医用画像技術についての知識やスキルを深めることができ、同様の分野を志す学生の方々や現場で活躍される社員さんの方々とお話しさせていただきました。
そして、景品としてカメラも贈呈いただき本当にありがとうございました。大切に使用させていただきます。


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
