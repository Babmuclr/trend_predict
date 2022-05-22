# 予測の詳細
論文数の予測の上位(予測値>400)の中で、実際の論文数が上位のものを成功例、実際の論文数が50以下のペアを失敗例とする。予測自体は、うまくいっているが、キーワードが類似しているものも失敗例にしている。

## 成功例
['design' 'performance']<br/>
['design' 'algorithms']<br/>
['theory' 'algorithms']<br/>
['optimization' 'linear programming']<br/>
['wireless communication' 'interference']<br/>
['sociology' 'statistics']<br/>
['visualization' 'feature extraction']<br/>
['remote sensing' 'satellites']<br/>
['interference' 'signal to noise ratio']<br/>
['switches' 'capacitors']<br/>

## 失敗例
1. 単純に予測がうまくいっていない<br/>
['design' 'human factors']<br/>
['design' 'experimentation']<br/>
['noise' 'estimation']
2. 同じようなキーワードが存在している<br/>
['wireless communication' 'wireless sensor networks']<br/>
['data models' 'computational modeling']<br/>
['mathematical model' 'computational modeling']<br/>
['mobile computing' 'mobile communication']<br/>
['wireless communication' 'wireless sensor networks']<br/>
['fpga' 'field programmable gate arrays']<br/>
['algorithm design' 'algorithm design and analysis']<br/>
['learning artificial intelligence' 'learning (artificial intelligenc']<br/>
['ofdm modulation' 'ofdm']<br/>

# 考察
研究トピックの選び方が不適切であることがわかる。トピックのペアが、類似関係や包含関係であるペアが、予測上位にきていることから、実験自体があまりうまくいっていないように考えられる。