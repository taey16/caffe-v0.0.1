
import random
import numpy as np

HEADER = '<head>\n \
<meta http-equiv=\"Content-Type\" content=\"text/html; charset=utf-8\">\n \
<script src=\"http://code.jquery.com/jquery-latest.js\"></script>\n \
<script type=\"text/javascript\" src=\"/static/js/dough-script-0.2.0.js\"> </script>\n \
<script src=\"http://cdnjs.cloudflare.com/ajax/libs/jquery-cookie/1.4.0/jquery.cookie.js\"></script>\n \
</head>\n'

SCRIPT_MOUSE_OVER = '<script>\n \
  $(\".image_show\").mouseover(function(e) {\n \
    img_x = $(this).parent().offset().left + $(this).width();\n \
    img_y = $(this).parent().offset().top - $(this).height(); \
    var wr = Math.min(1.0, 800.0 / this.naturalWidth);\n \
    var hr = Math.min(1.0, 600.0 / this.naturalHeight);\n \
    var ratio = Math.min(wr, hr);\n \
    var h = this.naturalHeight * ratio;\n \
    var w = this.naturalWidth * ratio;\n \
    console.log(h + \" \" + w + \" \" + ratio);\n \
    console.log(this.naturalHeight + " " + this.naturalWidth);\n \
    if ( img_y + h > $(document).height() ) {\n \
      img_y = $(document).height() - h - 80;\n \
    }\n \
    else { }\n \
    if ( img_x + w > $(document).width() ) {\n \
      img_x -= img_x + w - $(document).width() + $(document).width() - img_x + $(this).width();\n \
      img_x -= 20;\n \
    }\n \
    else {\n \
      img_x += 20;\n \
    }\n \
    console.log(img_x + \" \" + img_y);\n \
    div_tag = $(\'<div>\').css({position: \'absolute\', left: img_x, top: img_y});\n \
    div_tag.attr(\'id\', \'img_div0404\');\n \
    img_tag = $(\'<img>\');\n \
    img_tag.attr(\'src\', this.src);\n \
    img_tag.attr(\'style\', \'border: 4px solid; border-color: #f00; max-width: 800px; max-height: 600px;\');\n \
    div_tag.append(img_tag);\n \
    $(this).parent().append(div_tag);\n \
  }).mouseout(function() {\n \
    $(\"#img_div0404\").remove();\n \
  });\n \
  $.urlParam = function(name){\n \
    var results = new RegExp(\'[\\?&]\' + name + \'=([^&#]*)\').exec(window.location.href);\n \
    if (results==null){\n \
      return null;\n \
    }\n \
    else{ return results[1] || 0; }\n \
  }\n \
  $(document).ready(function() {\n \
    var page_no = $.urlParam(\'page_no\');\n \
    if ( page_no == null ) {\n \
      var latest_href = $.cookie(\'latest_href\');\n \
      if ( latest_href != undefined ) {\n \
        window.location.href = latest_href;\n \
      }\n \
    }\n \
    else {\n \
      //$.cookie(\'latest_href\', window.location.href);\n \
    }\n \
  });\n \
</script>\n'

CATE_ID = 'oxbuild'

PJT_ROOT = '/works/METRIC/'
LOG_FILENAME = 'log_%s.log' % CATE_ID
URL_PREFIX = 'http://10.202.211.120:2596/PBrain/oxbuild/oxbuild_images/'
DATASET_ROOT = '/storage/oxbuild/'
DATASET_GT_FILENAME = '%s_gt.txt' % CATE_ID

def generate_html( results, gt ):
  fo = open('%s/%s.html' % (PJT_ROOT, LOG_FILENAME), 'w')
  fo.write('<html>\n')
  fo.write(HEADER)
  fo.write('<body>\n')
  fo.write('<table>\n')
  for q, r in results.iteritems():
    fo.write('<tr>\n')
    fo.write('<td>\n')
    fo.write('<img img class="image_show" src=\"%s/%s\" height=\"128\" width=\"128\"></br><font size=2>%s</font>' % (URL_PREFIX, q, q)) 
    fo.write('</td>\n')
    for path in r:
      fo.write('<td>\n')
      doc_path, distance = path.split('||')[0], path.split('||')[1]
      if doc_path in gt[q]: 
        fo.write('<img img class="image_show" src=\"%s/%s\" height=\"128\" width=\"128\" style="border: 4px solid #aB12c3">' % (URL_PREFIX, doc_path)) 
      else:
        fo.write('<img img class="image_show" src=\"%s/%s\" height=\"128\" width=\"128\">' % (URL_PREFIX, doc_path)) 
      fo.write('</br><font size=2>%s</font>' % distance )
      fo.write('</td>\n')
    fo.write('</tr>\n')
    fo.write('<tr><td>&nbsp</td></tr>\n')

  fo.write('</table>\n')
  fo.write(SCRIPT_MOUSE_OVER)
  fo.write('</body>\n')
  fo.write('</html>\n')
  fo.close()

gt = dict([[entry.strip().split(' ')[0], entry.strip().split(' ')] for entry in open('%s/%s' % (DATASET_ROOT, DATASET_GT_FILENAME), 'r')])

entries = [entry.strip() for entry in open('%s/%s' % (PJT_ROOT, LOG_FILENAME), 'r')]
import pdb; pdb.set_trace()

query = entries[0::4]; query = query[:-1]
ranked_list=entries[1::4]

"""
random_idx = np.random.permutation(len(query))
q, r = [], []
for n in range(len(query)):
  q.append(query[random_idx[n]])
  r.append(ranked_list[random_idx[n]])
query, ranked_list = q, r
"""

if len(query) <> len(ranked_list):
  print 'num query and num ranked_list mismatached'; exit(-1)

results = {}
for n, q in enumerate(query):
  if n == 70: break
  q = q.split(' ')[-1]
  short_list = ranked_list[n].split(' ')
  results[q] = short_list

import pdb; pdb.set_trace()
generate_html( results, gt )

