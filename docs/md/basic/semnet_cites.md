# 概要
Predicting research trends with semantic and neural networks with an application in quantum physics を引用した論文のメモ。重要そうな被引用論文のみを詳細に読み、メモを残す。

# まとめ

## 研究トレンド予測について
研究トレンドの予測における課題となっている点
- トピックの人気度の決め方
    - 引用数、投稿数で決定
- トピックの特徴ベクトルの埋め込み方法
    - グラフ構造を用いたり色々している、Feature Engineering　が重要
- トピックの新規性の決め方　→　推薦を決定する<br>
    新規性の高いものから推薦する　→　元論文では、グラフの次数や類似度で計算
    - 投稿数、それらの変化数で決定

<img src="../pic/model.jpg" alt="研究トレンドの予測流れ" title="研究トレンドの予測">

研究トレンドの予測問題は、トピックのある時点tの人気度を予測する問題として扱っていることが多い。この問題で、課題となっているのは、次である。
1. どのようにコンセプトを入力する特徴ベクトルに変化するか?
2. 目的変数となる、コンセプトのある時点での、人気度をどのように、定義するか？

元論文のように、共起ネットワークを用いて、論文はあったが、解き方に関するものであった。（トピックの特徴ベクトルへの埋め込み部分の改善や手法の解説など）ネットワークに関する改善や問題設定を改善することは、していないように見える。

## 推薦について
論文の新規性についての研究は、たくさんありそう。
- What is an emerging technology?<br>
https://www.sciencedirect.com/science/article/pii/S0048733315001031?via%3Dihub
- Identifying emerging topics in science and technology<br>
https://www.sciencedirect.com/science/article/pii/S0048733314000298?via%3Dihub
- A bibliometric model for identifying emerging research topics<br>
https://asistdl.onlinelibrary.wiley.com/doi/10.1002/asi.23930


# 論文のメモ

## Combining deep neural network and bibliometric indicator for emerging research topic prediction
ディープニューラルネットワークとビブリオメトリック指標の組み合わせによる新興研究トピック予測<br>
https://www.sciencedirect.com/science/article/abs/pii/S0306457321001072

（感想）問題形式が異なる。関連研究は参考になるかも。推薦部分を、もう少し深く読む必要がある。

トピックの人気度予測問題を解いた論文。
- 目標指標の予測を時系列予測問題として定義し、DNNで予測している。時系列的な要素を強めに考えている。
- トピック毎(トピック数 n)に、特徴量(次元 m)を自作して、入力としている。そして、n個のトピックごとに、人気度を決めて、現時点がtとした時、t+1の人気度を予測。
- コンセプトの新規性と継続性を測る、ビブリオメトリック指標を作って、推薦するコンセプトを決める。

## Dynamic Embedding-based Methods for Link Prediction in Machine Learning Semantic Network
機械学習セマンティックネットワークにおけるリンク予測のための動的埋め込みベースの方法<br>
https://ieeexplore.ieee.org/abstract/document/9672040

（感想）既存手法の解き方がわかる。

コンペの問題を解いてみた論文。
- 方法１：グラフ＋時系列から、Node2Vecを用いて1996年から2017年までの、それぞれの年でベクトル化し、Transforemerにいれて、予測している。結果は、うまくいっていない。（AUC　0.73）
- 方法２：Feature Engineeringした特徴量 + Node2Vecのベクトルを用いた特徴量を作成して、MLPに入れて予測する。最高スコア、AUCが0.902に到達。

## Potential index: Revealing the future impact of research topics based on current knowledge networks
ポテンシャルインデックス 現在の知識ネットワークから、研究テーマの将来的な影響力の明確化<br>
https://www.sciencedirect.com/science/article/abs/pii/S1751157721000365

（感想）被引用論文数的なもので、トピックの重要度を測る方法は参考になる。トピックとキーワードを区別しているけど、他の論文ではどうなっているか？知りたくなった。

Potential Indexという、知識グラフのcentrality（中心性）とnetwork entropyから算出できるノードの一指標を提案している。使うとうまく予測ができるらしい。

## Embedding technique and network analysis of scientific innovations emergence in an arXiv-based concept network
arXivベースの概念ネットワークにおける科学技術イノベーション創発の埋め込み手法とネットワーク分析<br>
https://ieeexplore.ieee.org/abstract/document/9204220

（感想）よくわからない。

複雑ネットワーク科学    (https://www.topo.hokudai.ac.jp/education/SpecialLecture/090501.pdf)<br>
セマンティックネットワークを解析した論文。論文が何を言いたいのか、ふんわりしているけど、重み付け有効性とノード埋め込みの有用性を示している論文。
- 期間ごとのネットワークを作成して、投稿された論文数・論文数に関する指標で、重みをつける。重みを閾値で足切りして、グラフを再構築。重みをもとに、強いリンクと弱いリンクをラベルする。**PyTorch-BigGraph**をつかって、コンセプトをベクトル化する。コサイン類似度を算出する。
- 似たコンセプト・似てないコンセプトは、両方とも、2013年の強いリンクは、2015年も強いリンク。