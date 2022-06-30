# 予測の詳細

論文数の予測の上位(予測値>400)の中で、実際の論文数が上位のものを成功例、実際の論文数が50以下のペアを失敗例とする。予測自体は、うまくいっているが、キーワードが類似しているものも失敗例にしている。

## 成功例

['design' 'performance']  
['design' 'algorithms']  
['theory' 'algorithms']  
['optimization' 'linear programming']  
['wireless communication' 'interference']  
['sociology' 'statistics']  
['visualization' 'feature extraction']  
['remote sensing' 'satellites']  
['interference' 'signal to noise ratio']  
['switches' 'capacitors']  

## 失敗例

1. 単純に予測がうまくいっていない  
['design' 'human factors']  
['design' 'experimentation']  
['noise' 'estimation']
2. 同じようなキーワードが存在している  
['wireless communication' 'wireless sensor networks']  
['data models' 'computational modeling']  
['mathematical model' 'computational modeling']  
['mobile computing' 'mobile communication']  
['wireless communication' 'wireless sensor networks']  
['fpga' 'field programmable gate arrays']  
['algorithm design' 'algorithm design and analysis']  
['learning artificial intelligence' 'learning (artificial intelligenc']  
['ofdm modulation' 'ofdm']  

## 考察

研究トピックの選び方が不適切であることがわかる。トピックのペアが、類似関係や包含関係であるペアが、予測上位にきていることから、実験自体があまりうまくいっていないように考えられる。
