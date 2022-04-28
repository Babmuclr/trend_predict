# Aminerデータセット
Citation Network Dataset: （DBLP Citation, ACM Citation network）<br>
https://www.aminer.org/citation

## データの属性
|Field Name|Field Type|Description|Example|
|:----|:----|:----|:----|
|id|string|paper ID|43e17f5b20f7dfbc07e8ac6e|
|title|string|paper title|Data mining: concepts and techniques|
|authors.name|string|author name|Jiawei Han|
|author.org|string|author affiliation|Department of Computer Science, University of Illinois at Urbana-Champaign|
|author.id|string|author ID|53f42f36dabfaedce54dcd0c|
|venue.id|string|paper venue ID|53e17f5b20f7dfbc07e8ac6e|
|venue.raw|string|paper venue name|Inteligencia Artificial, Revista Iberoamericana de Inteligencia Artificial|
|year|int|published year|2000|
|keywords|list of strings|keywords|["data mining", "structured data", "world wide web", "social network", "relational data"]|
|fos.name|string|paper fields of study|Web mining|
|fos.w|float|fields of study weight|0.659690857|
|references|list of strings|paper references|["4909282", "16018031", "16159250", "19838944", ...]|
|n_citation|int|citation number|40829|
|page_start|string|page start|11|
|page_end|string|page end|18|
|doc_type|string|paper type: journal, book title...|book|
|lang|string|detected language|en|
|publisher|string|publisher|Elsevier|
|volume|string|volume|10|
|issue|string|issue|29|
|issn|string|issn|0020-7136|
|isbn|string|isbn|1-55860-489-8|
|doi|string|doi|10.4114/ia.v10i29.873|
|pdf|string|pdf URL|//static.aminer.org/upload/pdf/1254/ 370/239/53e9ab9eb7602d970354a97e.pdf|
|url|list|external links|["http://dx.doi.org/10.4114/ia.v10i29.873", "http://polar.lsi.uned.es/revista/index.php/ia/ article/view/479"]|
|abstract|string|abstract|Our ability to generate...|
|indexed_abstract|dict|indexed abstract|{"IndexLength": 164, "InvertedIndex": {"Our": [0], "ability": [1], "to": [2, 7, ...]}}|

## キーワードの分析