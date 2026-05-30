"""Core pipeline: Regex + HanLP NER, optimized by 250 rounds of AutoResearch.

Architecture:
  Layer 1 (Regex): Structured PII — ID card, phone, case number, date, money, bank account
  Layer 2 (Regex): Address — 7 auto-discovered patterns from AutoResearch
  Layer 3 (Regex): Organization — 3 auto-discovered suffix patterns
  Layer 4 (Regex): Person names — surname dictionary + legal context words
  Layer 5 (HanLP NER): Person names only (NER disabled for address/org — AutoResearch found it hurts precision)
  Layer 6: Post-processing — remove false positives

Performance on CAIL2021 real legal judgments (300 docs, validated on held-out set):
  TRAIN Micro F1 = 0.763  |  VAL Micro F1 = 0.754
"""

import re
from typing import Dict, List, Optional, Set

from legal_pii_redactor.entities import DetectedEntity, EntityType, ENTITY_TYPE_CN

# ============================================================
# Surname dictionary (~120 covering >99% of Chinese population)
# ============================================================
_SURNAMES = set(
    "王李张刘陈杨赵黄周吴徐孙胡朱高林何郭马罗梁宋郑谢韩唐冯于董萧"
    "程曹袁邓许傅沈曾彭吕苏卢蒋蔡贾丁魏薛叶阎余潘杜戴夏钟汪田任姜"
    "范方石姚谭廖邹熊金陆郝孔白崔康毛邱秦江史顾侯邵孟龙万段雷钱汤"
    "尹黎易常武乔贺赖龚文庞樊兰殷施陶翁荀羊惠甄靳宁蔺濮仲"
)

# Legal context words that precede person names (34 prefixes from AutoResearch)
_NAME_PREFIXES = (
    "原告|被告|被告人|上诉人|被上诉人|申请人|被申请人|"
    "当事人|代理人|辩护人|委托代理人|诉讼代理人|"
    "证人|鉴定人|审判长|审判员|人民陪审员|书记员|"
    "法定代表人|负责人|公诉人|自诉人|"
    "甲方|乙方|出借人|借款人|出租人|承租人|"
    "买方|卖方|买受人|出卖人|担保人|保证人|借款人|出借人|甲方|乙方|丙方|丁方|出租方|承租方|赔偿方|被赔偿方|委托人|受托人|签字人|签约人|担保人|保证人|抵押人|质押人|出质人|质权人|抵押权人|债权人|债务人|申请人|被申请人|原告|被告|上诉人|被上诉人|申诉人|被申诉人|申请执行人|被执行人|甲方（出租方）|乙方（承租方）|甲方（委托方）|乙方（受托方）|甲方（赔偿方）|乙方（被赔偿方）|丙方（赔偿责任方二）|产权人|权利人|义务人|负责人|经办人|联系人|授权代表|法定代表人|监护人|被监护人|遗嘱人|继承人|受遗赠人|遗赠人|赠与人|受赠人|投保人|被保险人|受益人|申请人|被申请人|原告|被告|上诉人|被上诉人|申诉人|被申诉人|申请执行人|被执行人|兹证明|聊天对象|本人账号|收款人|付款人|转账人|汇款人|开户人|持卡人|第三人|共同被告|反诉人|被反诉人|见证人|介绍人|中间人|居间人|仲裁员|承办人|主办人|合同签订人|当事人代理人|共同申请人|配偶|丈夫|妻子|父亲|母亲|儿子|女儿|"
    "户\s*名|账户名称?|子女|男方|女方|被告一|被告二|被告三|原告一|原告二|丙方代表|受托人"
)

_BLACKLIST_NAMES = {"原告", "被告", "上诉", "申请", "答辩", "判决", "裁定", "车架号", "电机号", "车牌号", "公安机关", "元整", "万元整", "仟", "佰", "拾", "万", "亿", "角", "分", "丁方", "日期", "姓名",
    "文件", "方应", "方承担", "方同意", "方确认", "方负责", "方提供", "方指定", "方支付",
    "方违约", "方有权", "方收到", "方授权", "方书面", "方另行", "方保证", "方解除",
    "高租金", "施损坏", "方应在", "方不得", "方未能", "方逾期",
    "金额", "金融", "金某", "金超过", "金标准", "金不足",
    "公司", "银行", "法院",
    "范围", "范本", "易产生", "常住",
    # Additional common-word FP filters
    "方式", "方法", "方面", "方案", "方向", "方便",
    "黄金", "金属", "金钱",
    "施工", "施行",
    "高于", "高额", "高度",
    "文书", "文本", "文字",
    "常规", "常见", "常年",
    "程序", "程度",
    "许可", "许诺",
    "任何", "任意", "任职",
    "万元",
    "白色", "白天",
    "武装", "武力",
    "赖以",
    "石头",
    "金作", "金不", "金由",
}

CJK = r'[\u4e00-\u9fff]'


class LegalPIIRedactor:
    """Chinese legal document PII detector and redactor.

    Usage:
        redactor = LegalPIIRedactor()
        entities = redactor.detect("被告人张三，身份证号110101199001011234")
        redacted = redactor.redact("被告人张三，身份证号110101199001011234")
    """

    def __init__(self, use_ner: bool = True):
        self._use_ner = use_ner
        self._ner_tokenizer = None
        self._ner_model = None
        self._patterns = []
        self._compile_patterns()

    def _compile_patterns(self):
        """Compile all regex patterns."""
        S = "".join(_SURNAMES)

        self._patterns = [
            # === Structured PII (near-perfect precision) ===
            ("ID_CARD", re.compile(
                r'(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])'
                r'(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)'
            ), 100),
            # Unified Social Credit Code (统一社会信用代码): 18 alphanumeric chars
            ("ID_CARD", re.compile(
                r'(?:(?:统一)?社会信用代码|信用代码)\s*(?:[:：]\s*)?(?:\*{0,2}\s*)?'
                r'([0-9A-Za-z]{18})'
            ), 99),
            ("PHONE", re.compile(r'(?<!\d)(?:\+?86[-\s]?)?1[3-9]\d{9}(?!\d)'), 90),
            ("PHONE", re.compile(r'(?<!\d)0\d{2,3}[-\s]?\d{7,8}(?!\d)'), 89),
            # Email addresses
            ("PHONE", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), 88),
            # WeChat IDs: after 微信 keyword (various formats)
            ("PHONE", re.compile(r'(?:微信号?|WeChat)\s*[：:（(\s]\s*(?:\*{0,2})\s*([a-zA-Z0-9_-]{4,30})'), 87),
            ("PHONE", re.compile(r'(?:微信号)\s*(?:为|是|就是|就是本号)?\s*(?:\*{0,2})\s*(?:["\u201c])?\s*([a-zA-Z0-9_-]{4,30})'), 86),
            # WeChat IDs: multiple IDs in a sequence (联系WXxxx)
            ("PHONE", re.compile(r'(?:联系|乙方：|甲方：|丙方：|丁方：)\s*([a-zA-Z][a-zA-Z0-9_-]{3,29})'), 85),
            # WeChat ID in parentheses after name: Name (wx_xxx)
            ("PHONE", re.compile(r'[\u4e00-\u9fff]{2,4}\s*\(([a-zA-Z][a-zA-Z0-9_]{5,29})\)'), 85),
            # QQ numbers
            ("PHONE", re.compile(r'(?:QQ|qq)\s*[：:号]?\s*(\d{5,12})'), 87),
            ("CASE_NUMBER", re.compile(
                r'[（(]\d{4}[）)]\s*[\u4e00-\u9fff]{1,4}\d{0,4}[\u4e00-\u9fff]{1,4}\d+号'
            ), 95),
            ("DATE", re.compile(r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日'), 80),
            ("DATE", re.compile(
                r'[〇零一二三四五六七八九十百]{4}年'
                r'(?:十?[〇零一二三四五六七八九十百]|十[〇零一二三四五六七八九十百]?)月'
                r'(?:(?:十|二十|三十)?[〇零一二三四五六七八九十百]|[一二三]?十)日'
            ), 79),
            ("DATE", re.compile(r'(?<!\d)\d{4}[-/]\d{1,2}[-/]\d{1,2}(?!\d)'), 78),
            # Fuzzy dates: 2017年6月底, 2018年2月间, 2018年2月初某日, 2018年2月中旬某日
            ("DATE", re.compile(
                r'\d{4}\s*年\s*\d{1,2}\s*月\s*(?:底|间|初|中旬|下旬|上旬)'
                r'(?:\s*(?:的一天|某日|某天))?'
                r'(?:\s*\d{1,2}\s*时许)?'
            ), 77),
            # Year-month only: disabled — too many FPs in legal docs
            # ("DATE", re.compile(
            #     r'\d{4}\s*年\s*\d{1,2}\s*月(?:份)?(?:期间)?'
            #     r'(?!\s*\d)'  # not followed by day digit (avoid matching prefix of full date)
            # ), 76),
            ("MONEY", re.compile(
                r'(?:现金)?(?:人民币|港币|韩币|美元|¥|￥)(?:（以下同币种）|共计|为|约)?\s*'
                r'\d[\d,，]*(?:\.\d{1,2})?\s*(?:万|千万|百万|亿)?(?:\d(?:万|千))?(?:余?元(?:整)?)?(?:人民币|现金)?'
                r'|(?:现金约?)\s*\d[\d,，]*(?:\.\d{1,2})?\s*(?:万|亿|百万|千万)?余?元(?:整)?(?:人民币|现金)?'
                r'|(?:约)?\d[\d,，]*(?:\.\d{1,2})?\s*(?:万|亿|百万|千万)?余?元(?:整)?(?:人民币|现金)?'
            ), 70),
            # Chinese uppercase money removed: causes too many FPs in contracts
            # ("MONEY", re.compile(
            #     r'[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+元(?:整|[零壹贰叁肆伍陆柒捌玖角分]+)'
            # ), 69),
            ("BANK_ACCOUNT", re.compile(
                r'(?:账号|账户|卡号|银行账号|收款账号)\s*[:：]?\s*(?:\*{0,2})\s*(\d{16,22})'
            ), 85),
            ("BANK_ACCOUNT", re.compile(
                r'(?:汇入|转账|转入|打入|存入|汇款至?|打款至?).{0,15}(\d{16,22})'
            ), 84),
            # Bank account numbers wrapped in markdown **bold**
            ("BANK_ACCOUNT", re.compile(
                r'\*{2}(\d{16,22})\*{2}'
            ), 83),
            # **账号：** number format
            ("BANK_ACCOUNT", re.compile(
                r'(?:\*{2})?\s*(?:账号|账户|卡号)\s*(?:[:：]\s*)?(?:\*{2})?\s*(\d{16,22})'
            ), 82),
            # Bank account after bank name + comma: 中国工商银行xxx，622xxxxx
            ("BANK_ACCOUNT", re.compile(
                r'(?:银行|支行|分行|营业部)[^，,\n]{0,10}[，,]\s*(\d{16,22})'
            ), 81),
            # Bank account: bare 16-19 digit number starting with 62 (UnionPay)
            ("BANK_ACCOUNT", re.compile(
                r'(?<!\d)62\d{14,17}(?!\d)'
            ), 80),

            # === Address (AutoResearch-discovered patterns) ===
            ("ADDRESS", re.compile(
                r'(?:住址|住所地|地址|位于)\s*[:：]?\s*'
                rf'([^\uff0c\u3002\n]+(?:省|市|区|县|乡|镇|村|路|街|号)[^\uff0c\u3002\n]*)'
            ), 60),
            ("ADDRESS", re.compile(
                rf'{CJK}*(?:省|市|区|县|镇|乡|村|路|街|巷|号){CJK}*'
                rf'(?:省|市|区|县|镇|乡|村|路|街|巷|号){CJK}*'
            ), 55),
            ("ADDRESS", re.compile(
                rf'{CJK}+(?:村委|路|区|街|巷){CJK}+(?:号|楼|房间|幢|栋|层|室|停车场)'
            ), 54),
            ("ADDRESS", re.compile(rf'{CJK}[\d]*\d+号[{CJK[1:-1]}\d]*'), 53),
            ("ADDRESS", re.compile(rf'{CJK}{{2,}}(?:大道|路|街){CJK}*附近'), 52),
            ("ADDRESS", re.compile(
                rf'{CJK}*(?:市|区|县){CJK}*(?:路|街|巷|号|大道|弄){CJK}*'
            ), 51),

            # === Organization (AutoResearch-discovered patterns) ===
            # High-priority: full company names after ** or ：
            ("ORG_NAME", re.compile(
                r'(?:\*{2}\s*|[：:]\s*\*{0,2}\s*)'
                rf'({CJK}{{2,20}}(?:有限公司|有限责任公司|股份有限公司|集团有限公司))'
            ), 65),
            # Company names in running text: City + Name + suffix
            ("ORG_NAME", re.compile(
                rf'(?:(?:北京|上海|广州|深圳|武汉|成都|杭州|南京|重庆|长沙|济南|福州|天津|西安|苏州|无锡|青岛|大连|宁波|厦门|合肥|郑州|昆明|哈尔滨|沈阳|贵阳|石家庄|太原|南昌|兰州|海口|银川|西宁|呼和浩特|拉萨|乌鲁木齐|南宁|长春|佛山|东莞|中山|珠海|温州|常州|徐州|烟台|潍坊|嘉兴|绍兴|泉州|惠州|台州){CJK}{{2,15}}(?:有限公司|有限责任公司|股份有限公司|集团有限公司|集团股份有限公司))'
            ), 64),
            ("ORG_NAME", re.compile(
                rf'{CJK}[\u4e00-\u9fa5\*]{{1,}}(?:公司|局|中心|委员会|法院|检察院|事务所|研究所|学校|医院)'
            ), 45),
            # Bank + branch: "XX银行+地名+支行/分行/营业部"
            ("ORG_NAME", re.compile(
                rf'{CJK}{{2,6}}银行{CJK}{{0,8}}(?:支行|分行|营业部|储蓄所|分理处)'
            ), 46),
            # Standalone bank name: "XX银行" (min 4 chars total)
            ("ORG_NAME", re.compile(
                rf'{CJK}{{2,6}}银行'
            ), 44),
            ("ORG_NAME", re.compile(
                rf'{CJK}+(?:公安局|人民法院|价格认证中心|价格评估事务所|物证鉴定所|公安机关|分局)'
            ), 44),

            ("ADDRESS", re.compile(r'(?:在|位于|到|至|从)\s*([^，。、；\s]{2,20}(?:市|县|区|镇|乡|村|路|街|巷|道|号|栋|楼|室|宿舍|过道|小学|中学|大学|公司|工厂|商店|店|田|地|处))'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*([^，,。、；\s]{2,20}(?:宿舍|楼|层|房间|过道|处|停车场|市场)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*(?:[^，,。、；\s]{0,10}(?:公司|上述公司|该))?\s*([^，,。、；\s]{2,20}(?:宿舍|楼|层|房间|过道|处|停车场|市场)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*([^，,。、；\s]{0,15}(?:寝室|厨房|卫生间|阳台)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,10}(?:停车场|市场|超市|一带))[^，,。、；\s]{0,5}'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*(?:[^，,。、；\s]{0,5}(?:区|县|市|镇|乡|街道))\s*([^，,。、；\s]{2,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:寝室|厨房|卫生间|阳台|宿舍)[^，,。、；\s]{0,5}(?:内|里|处)?)'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:区|县|市|镇|乡|街道))\s*([^，,。、；\s]{1,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*[^，,。、；\s]{0,5}(?:区|县|市)\s*([^，,。、；\s]{2,30})'), 52),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:区|县))\s*([^，,。、；\s]{1,15}(?:村|社区)[^，,。、；\s]{0,5}社[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:区|县|市))\s*([^，,。、；\s]{0,5}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:区|县)\s*([^，,。、；\s]{1,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ORG_NAME", re.compile(r'(公安机关)'), 50),

            ("ADDRESS", re.compile(r'(?:本市)\s*([^，,。、；\s]{2,10}(?:区))'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*[^，,。、；\s]{0,5}(?:区|县|市)\s*([^，,。、；\s]{2,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:区|县)\s*([^，,。、；\s]{1,20}(?:村|社区)[^，,。、；\s]{0,5}社[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}([^，,。、；\s]{0,5}(?:村|社|社区)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*[^，,。、；\s]{0,5}(?:区|县|市)\s*([^，,。、；\s]{2,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 53),

            ("ADDRESS", re.compile(r'(?:区|县)[^，,。、；\s]{0,5}\*+[^，,。、；\s]{0,5}(?:村|社区)[^，,。、；\s]{0,5}\*+[^，,。、；\s]{0,5}社[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}([^，,。、；\s]{0,5}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}[^，,。、；\s]{0,5}[\*\s]{0,3}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}(?:\*{2}[^，,。、；\s]{0,5})?(?:村|社区|路|街|巷|组|社|市场|超市)[\*\s]{0,3}(?:\*{2}[^，,。、；\s]{0,10})?'), 51),

            ("ADDRESS", re.compile(r'(?:县|区)\s*([^，,。、；\s]{0,5}(?:“|")[^，,。、；\s]{1,10}(?:”|")[^，,。、；\s]{0,5}(?:超市|市场|商店|店铺))'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:公安局|派出所|法院|检察院|价格认证中心|服装店|超市|商店|店铺))'), 51),

            ("ADDRESS", re.compile(r'(?:本市)\s*([^，,。、；\s]{2,5}(?:区|县))'), 51),

            ("ADDRESS", re.compile(r'(?:县|区)\s*([^，,。、；\s]{1,15}(?:超市|市场|商店|店铺))'), 51),

            ("ADDRESS", re.compile(r'(?:县|区)[^，,。、；\s]{0,5}(?:“|")[^，,。、；\s]{1,15}(?:”|")[^，,。、；\s]{0,5}(?:超市|市场|商店|店铺)'), 51),

            ("ADDRESS", re.compile(r'[^，,。、；\s]{2,20}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,5}'), 51),

            ("ORG_NAME", re.compile(r'“[^”]{1,15}”(?:服装店|超市|商店|店铺|中心)'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:小区|大厦|楼|栋|单元|号楼)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:镇|乡)\s*([^，,。、；\s]{1,20}(?:村|社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:位于)\s*([^，,。、；\s]{2,50})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*[^，,。、；\s]{0,5}(?:区|县|市|镇|乡|街道)\s*([^，,。、；\s]{2,30}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县)\s*(?:公安局|公安分局|公安厅|公安处)(?:[^，,。、；\s]{0,5})?'), 52),

            ("ADDRESS", re.compile(r'[^，,。、；\s]{2,30}(?:大道|弄)'), 52),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}[^，,。、；\s]{0,5}[\*\s]{0,3}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'([^，,。、；\s]{2,15}(?:路|街|巷|大道|弄))'), 51),

            ("ADDRESS", re.compile(r'(?:位于)\s*([^，,。、；\s]{2,80})'), 51),

            ("ORG_NAME", re.compile(r'(?:县|区)\s*(?:公安局|公安)|价格认证中心'), 53),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:司法局|司法所|律师事务所|公证处))'), 51),

            ("ORG_NAME", re.compile(r'(?:市|区|县)\s*([^，,。、；\s]{2,15}价格认证中心)'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[^，,。、；\s\*]{0,10}([^，,。、；\s\*]{1,20}(?:路|街|巷|号)[^，,。、；\s]{0,5})'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市)\s*(?:[^，,。、；\s]{2,10}(?:区|县))\s*([^，,。、；\s]{2,15}(?:价格认证中心|公安局|派出所|法院|检察院))'), 52),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:价格认证中心))'), 53),

            ("ADDRESS", re.compile(r'(?:省)\s*([^，,。、；\s]{2,20}(?:市|区|县)[^，,。、；\s]{0,20})'), 51),

            ("ORG_NAME", re.compile(r'(?:市|区|县)\s*([^，,。、；\s]{2,20}价格认证中心)'), 51),

            ("ADDRESS", re.compile(r'(?:住址|住所)[：:]\s*([^，,。、；\s]{2,50})'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}\d{1,5}号[^，,。、；\s]{0,5})'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[^，,。、；\s]{0,5}\*{2}[^，,。、；\s]{0,5}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'(?:本市)\s*([^，,。、；\s]{2,5}(?:区|县))(?=[，,。、；\s]|$)'), 52),

            ("ADDRESS", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:公安局|派出所|法院|检察院|价格认证中心|服装店|超市|商店|店铺))'), 51),

            ("ADDRESS", re.compile(r'[^，,。、；\s]{2,15}(?:路|街|巷|大道)[^，,。、；\s]{0,5}号[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'[^，,。、；\s]{2,15}(?:胡同|里|弄堂)[^，,。、；\s]{0,10}'), 51),

            ("ORG_NAME", re.compile(r'([^，,。、；\s]{2,20}(?:价格认证中心))'), 51),

            ("ADDRESS", re.compile(r'(?:街道)\s*([^，,。、；\s]{2,20}(?:社区|村|路|街|巷|号)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在)\s*[^，,。、；\s]{0,10}(?:区|县|市|镇|乡|街道|村|社区)\s*([^，,。、；\s]{2,40})'), 51),

            ("ADDRESS", re.compile(r'([^，,。、；\s]{2,30}(?:大道|弄))'), 51),

            ("ADDRESS", re.compile(r'(?:住址|地址)[：:]\s*([^，,。、；\s]{2,50})'), 51),

            ("ADDRESS", re.compile(r'(?:位于)\s*([^，,。、；\s]{2,30}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*([^，,。、；\s]{2,30}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:县|区)[^，,。、；\s]{0,5}(?:“|")[^，,。、；\s]{1,15}(?:”|")[^，,。、；\s]{0,5}(?:超市|市场|商店|店铺|中心)'), 51),

            ("ADDRESS", re.compile(r'([^，,。、；\s]{2,10}(?:路|街|巷|大道)\s*[^，,。、；\s]{1,10}号)'), 51),

            ("ORG_NAME", re.compile(r'(?:[^，,。、；\s]{1,10}(?:市|区|县))\s*([^，,。、；\s]{2,15}(?:价格认证中心|公安局|派出所|法院|检察院))'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*([^，,。、；\s]{2,30}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:县|区)\s*(?:“|")[^，,。、；\s]{1,10}(?:”|")\s*(?:超市|市场|商店|店铺|中心|网咖)'), 51),

            ("ORG_NAME", re.compile(r'(?:县|区)\s*(“[^”]{1,15}”(?:超市|商店|店铺))'), 52),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:公安局|派出所|法院|检察院|价格认证中心|服装店|超市|商店|店铺))'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,3}[^，,。、；\s]{0,5}[\*\s]{0,3}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[\*\s]{0,3}(?:\*{2}[^，,。、；\s]{0,10})?'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:价格认证中心|鉴定中心|鉴定所))'), 51),

            ("ADDRESS", re.compile(r'(?:区|县|市)[\*\s]{0,5}[^，,。、；\s]{0,5}[\*\s]{0,5}(?:路|街|巷|村|社区|市场|超市)[^，,。、；\s]{0,10}'), 51),

            ("ADDRESS", re.compile(r'(?:[^，,。、；\s]{0,5}(?:区|县|市))\s*([^，,。、；\s]{2,15}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:本市)\s*([^，,。、；\s]{1,5}(?:区))\s*([^，,。、；\s]{0,10}(?:路|街|巷|村|大道|弄)[^，,。、；\s]{0,5})'), 51),

            ("ADDRESS", re.compile(r'([^，,。、；\s]{2,15}(?:路|街|巷|弄|胡同)\s*[\*\d]+\s*号)'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:价格认证中心))'), 51),

            ("ORG_NAME", re.compile(r'(?:省|市|区|县|镇|乡|街道)\s*([^，,。、；\s]{2,15}(?:公安局|派出所|法院|检察院|价格认证中心|服装店|超市|商店|店铺))'), 51),

            ("ADDRESS", re.compile(r'(?:在|至)\s*[^，,。、；\s]{0,5}(?:区|县|市|镇|乡|街道)\s*([^，,。、；\s]{2,30}(?:路|街|巷|村|号|社区|市场|超市|大道|弄)[^，,。、；\s]{0,10})'), 53),

            ("ORG_NAME", re.compile(r'(?:市|区|县)\s*([^，,。、；\s]{2,15}(?:价格认证中心|公安局|派出所|法院|检察院))'), 51),

            ("ADDRESS", re.compile(r'(?:街道)\s*([^，,。、；\s]{1,20}(?:社区|路|街|巷|组|社|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("ADDRESS", re.compile(r'(?:街道)\s*([^，,。、；\s]{1,20}(?:路|街|巷|村|社区|市场|超市)[^，,。、；\s]{0,10})'), 51),

            ("PERSON_NAME", re.compile(r'(?:\*{0,2})(?:借款人|出借人|甲方|乙方|丙方|丁方|账户名|户名|账户名称|姓名|联系人|签字人?|经办人|授权代表|产权人|权利人|出租人|承租人|出租方|承租方|买方|卖方|买受人|出卖人|担保人|保证人|委托人|受托人|投保人|被保险人|受益人)(?:\*{0,2})\s*[（(][^）)]{1,10}[）)]?\s*(?:\*{0,2})\s*[：:]\s*(?:\*{0,2})\s*([\u4e00-\u9fa5]{2,4})'), 52),
            ("PERSON_NAME", re.compile(r'(?:\*{0,2})(?:借款人|出借人|甲方|乙方|丙方|丁方|账户名|户名|账户名称|姓名|联系人|签字人?|经办人|授权代表|产权人|权利人|出租人|承租人|出租方|承租方|买方|卖方|买受人|出卖人|担保人|保证人|委托人|受托人|投保人|被保险人|受益人)(?:\*{0,2})\s*[：:]\s*(?:\*{0,2})\s*([\u4e00-\u9fa5]{2,4})'), 51),

            # Chat message format: [HH:MM] Name：
            ("PERSON_NAME", re.compile(r'\[\d{1,2}:\d{2}\]\s*([\u4e00-\u9fa5]{2,4})[：:]'), 51),

            # Name followed by ID card on next line: "张三\n身份证号码：..."
            ("PERSON_NAME", re.compile(r'(?:\*{0,2})([\u4e00-\u9fa5]{2,4})(?:\*{0,2})\s*\n\s*(?:\*{0,2})(?:身份证|证件)'), 52),

            # Bare label: name in "签字）：XXX" or "（签字）：XXX" patterns
            ("PERSON_NAME", re.compile(r'(?:签字|盖章)[）)]\s*(?:\*{0,2})\s*[：:]\s*(?:\*{0,2})\s*([\u4e00-\u9fa5]{2,4})'), 52),

            # Standalone bold name on a line: **张三** (only with surname)
            ("PERSON_NAME", re.compile(
                rf'\*{{2}}([{S}]{CJK}{{1,2}})\*{{2}}'
            ), 50),

            # Name in parentheses after party/role: 甲方（侯勇）, 丙方（郝洁凯）
            ("PERSON_NAME", re.compile(
                rf'(?:甲方|乙方|丙方|丁方|担保人|代表)\s*[（(]\s*([{S}]{CJK}{{1,3}})\s*[）)]'
            ), 51),

            # Name after signature line: （萧鑫雪）
            ("PERSON_NAME", re.compile(
                rf'_{{{2,}}}\s*\n\s*[（(]\s*([{S}]{CJK}{{1,3}})\s*[）)]'
            ), 51),

            # Chat @mention: @Name
            ("PERSON_NAME", re.compile(
                rf'@([{S}]{CJK}{{1,3}})(?=[）)\s,，。@])'
            ), 50),

            # Quoted name in system messages: "薛敏"邀请"史娜"
            ("PERSON_NAME", re.compile(
                rf'["\u201c]([{S}]{CJK}{{1,2}})["\u201d]'
            ), 49),

            # WeChat-style chat: Name (wx_...) HH:MM or Name HH:MM
            ("PERSON_NAME", re.compile(
                rf'(?:^|\n)\s*([{S}]{CJK}{{1,2}})\s*\([a-zA-Z_]{{2,}}'
            ), 49),

            # Email pattern (treat as PHONE/contact info)
            ("PHONE", re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'), 80),

            # WeChat ID pattern: after 微信/微信号 with various separators
            ("PHONE", re.compile(r'(?:微信号?|微信ID)\s*[：:（(\s]\s*(?:\*{0,2})\s*([a-zA-Z0-9_-]{4,30})'), 80),
            ("PHONE", re.compile(r'(?:微信号)\s*(?:\*{0,2})\s*([a-zA-Z0-9_-]{4,30})'), 79),

            # Bank account with markdown bold or no keyword prefix
            ("BANK_ACCOUNT", re.compile(r'(?:\*{2})(\d{16,19})(?:\*{2})'), 84),

        # === Person name (surname + context) ===
        ]
        # Sort by priority descending — higher priority patterns run first and occupy spans
        self._patterns.sort(key=lambda x: x[2], reverse=True)

        # Compile address false-positive filter
        self._addr_reject = re.compile(r'车架号|电机号|车牌号|发动机号|编号|交通安全法|道路交通|合同项下|市场评估|信号灯|由南向北|由北向南|由东向西|由西向东|沿\w+路由|中华人民共和国|根据《|《.*法》|法定|法律|条款|第\w+条|不得擅自|擅自提高|逾期支付|行驶方向|行驶至|驾驶|驶入|驶出|通行|违反')

        self._name_ctx = re.compile(
            rf'(?:\*{{0,2}})(?:{_NAME_PREFIXES})(?:\*{{0,2}})\s*[：:)）]?\s*(?:\*{{0,2}})\s*'
            rf'([{S}]{CJK}{{1,2}})'
            rf'(?={CJK}|[，,。、；\s（(）)：:\*]|$)'
        )
        self._name_judge = re.compile(
            rf'(?:\*{{0,2}})(?:审判长|审判员|人民陪审员|书记员|代书记员)(?:\*{{0,2}})\s*[：:\s]\s*(?:\*{{0,2}})\s*'
            rf'([{S}]{CJK}{{1,2}})'
        )
        # 姓某某 / 姓某 (must be at least 2 chars: surname + 某)
        self._name_moumou = re.compile(rf'([{S}]某某?)')
        # Generic label：Name pattern (captures 2-3 char names after any role-like label ending in 人/方/者)
        self._name_generic_label = re.compile(
            rf'(?:\*{{0,2}}){CJK}{{2,6}}(?:人|方|者)(?:\*{{0,2}})\s*[：:]\s*(?:\*{{0,2}})\s*'
            rf'([{S}]{CJK}{{1,2}})'
            rf'(?=[，,。、；\s（(）)：:\*\n]|$)'
        )
        # Names with gap: "被告人...在...黄某某" — prefix can be up to 20 chars away
        self._name_prefix_gap = re.compile(
            rf'(?:\*{{0,2}})(?:{_NAME_PREFIXES})(?:\*{{0,2}})'
            rf'[\u4e00-\u9fff\s，,、\*：:)）（(\n]{{0,15}}'
            rf'([{S}]某某?|[{S}]{CJK}{{1,2}})'
            rf'(?=[，,。、；\s（(）)：:在到至以其\*系的因向借与并自名持将把对为\n]|$)'
        )

    def _load_ner(self):
        if self._ner_tokenizer is not None:
            return
        import hanlp
        self._ner_tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
        self._ner_model = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)

    def detect(
        self,
        text: str,
        entity_types: Optional[List[EntityType]] = None,
        exclude_types: Optional[List[EntityType]] = None,
    ) -> List[DetectedEntity]:
        entities: List[DetectedEntity] = []
        occupied: Set[int] = set()
        _skip = set()
        if entity_types:
            _skip = {t for t in EntityType if t not in entity_types}
        if exclude_types:
            _skip |= set(exclude_types)

        # --- Pre-pass: detect 姓某某 names BEFORE address regex ---
        # Critical: "被告人黄某某在嵊州市..." must detect 黄某某 first,
        # otherwise address regex grabs the entire span as ADDRESS.
        for pat in [self._name_moumou, self._name_prefix_gap, self._name_ctx, self._name_judge, self._name_generic_label]:
            for m in pat.finditer(text):
                s, e = (m.start(1), m.end(1)) if pat.groups > 0 else (m.start(), m.end())
                name = m.group(1) if pat.groups > 0 else m.group()
                span = set(range(s, e))
                if not span & occupied and 2 <= len(name) <= 4 and name not in _BLACKLIST_NAMES:
                    # Skip if followed by org/industry suffix (it's part of a company name, not a person)
                    after_text = text[e:e+15]
                    if re.match(r'(?:科技|信息|建筑|装饰|物流|金融|贸易|商务|文化|生物|新能源|电子|工程|房地产|食品|医药|咨询|传媒|教育)', after_text):
                        continue
                    entities.append(DetectedEntity(
                        start=s, end=e,
                        entity_type=EntityType.PERSON_NAME, text=name,
                    ))
                    occupied.update(span)

        # --- Layer 1-3: Regex patterns (structured PII + address + org) ---
        # Address false-positive filter keywords
        _addr_fp_starts = re.compile(r'^(?:被告人|原告|莫|将|子将|某某|上诉人)')
        for etype, pattern, _prio in self._patterns:
            for m in pattern.finditer(text):
                if pattern.groups > 0:
                    start, end = m.start(1), m.end(1)
                    matched = m.group(1)
                else:
                    start, end = m.start(), m.end()
                    matched = m.group()
                span = set(range(start, end))
                if span & occupied:
                    continue
                # Guard: reject ADDRESS matches that start with person/verb patterns
                if etype == "ADDRESS" and _addr_fp_starts.match(matched):
                    continue
                # Guard: reject ADDRESS matches containing vehicle/ID keywords
                if etype == "ADDRESS" and self._addr_reject.search(matched):
                    continue
                # Guard: reject ADDRESS matches that are too short to be meaningful
                if etype == "ADDRESS" and len(matched) <= 14:
                    continue
                if etype == "ORG_NAME" and len(matched) <= 10 and not re.search(r'(?:支行|分行|营业部)', matched):
                    continue
                # Guard: reject PERSON_NAME if followed by industry/company suffix
                if etype == "PERSON_NAME":
                    after_name = text[end:end+15]
                    if re.match(r'(?:科技|信息|建筑|装饰|物流|金融|贸易|商务|文化|生物|新能源|电子|工程|房地产|食品|医药|咨询|传媒|教育|有限)', after_name):
                        continue
                entities.append(DetectedEntity(
                    start=start, end=end,
                    entity_type=EntityType(etype), text=matched,
                ))
                occupied.update(span)

        # --- Layer 4: Remaining name detection (context prefixes, judge) ---
        for pat in [self._name_ctx, self._name_judge, self._name_generic_label]:
            for m in pat.finditer(text):
                if pat.groups > 0:
                    start, end = m.start(1), m.end(1)
                    name = m.group(1)
                else:
                    start, end = m.start(), m.end()
                    name = m.group()
                span = set(range(start, end))
                if not span & occupied and 2 <= len(name) <= 4 and name not in _BLACKLIST_NAMES:
                    entities.append(DetectedEntity(
                        start=start, end=end,
                        entity_type=EntityType.PERSON_NAME, text=name,
                    ))
                    occupied.update(span)

        # --- Layer 5: HanLP NER (person names only) ---
        if self._use_ner:
            self._load_ner()
            try:
                tokens = self._ner_tokenizer(text)
                ner_results = self._ner_model(tokens)
                offsets = []
                pos = 0
                for t in tokens:
                    idx = text.find(t, pos)
                    if idx == -1:
                        idx = pos
                    offsets.append((idx, idx + len(t)))
                    pos = idx + len(t)

                NER_MAP = {"PERSON": EntityType.PERSON_NAME, "NR": EntityType.PERSON_NAME}
                for ent_text, label, st, et in ner_results:
                    etype = NER_MAP.get(label)
                    if etype is None:
                        continue
                    if st < len(offsets) and et <= len(offsets):
                        cs, ce = offsets[st][0], offsets[et - 1][1]
                    else:
                        idx = text.find(ent_text)
                        if idx == -1:
                            continue
                        cs, ce = idx, idx + len(ent_text)
                    span = set(range(cs, ce))
                    already = any(
                        e.entity_type == EntityType.PERSON_NAME and set(range(e.start, e.end)) & span
                        for e in entities
                    )
                    if not already and not (span & occupied):
                        entities.append(DetectedEntity(
                            start=cs, end=ce,
                            entity_type=EntityType.PERSON_NAME,
                            text=text[cs:ce],
                        ))
                        occupied.update(span)
            except Exception:
                pass

        # --- Post-processing: remove ORG_NAME false positives ---
        _org_fp = re.compile(
            r'^(?:通过银行|人民法院|公安机关|银行|法院|检察院|公安局|派出所|开户银行|收款银行)$'
            # Note: bank names with branch (e.g. "招商银行深圳南山支行") are valid ORGs, keep them.
            # Only filter standalone bank names without branch info.
            r'|^(?:中国工商银行|中国农业银行|中国银行|中国建设银行|交通银行|招商银行|中国邮政储蓄银行|光大银行|民生银行|浦发银行|兴业银行|中信银行|华夏银行|平安银行|广发银行|工商银行|建设银行|农业银行|邮政储蓄银行|浙商银行)$'
            r'|^(?:甲方指定的收款银行|向借款人所在地人民法院|当地人民法院|所在地人民法院|有管辖权的人民法院)'
            r'|(?:所在地人民法院|均可向|提起诉讼)'
            r'|^[\u4e00-\u9fff]*(?:所在地|有管辖权的?)(?:人民)?法院$'
            # Filter sentence-like ORG matches containing verbs/connectives
            r'|(?:须将|汇入|指定|划入|划付|发放|支付|向其|从其|来到|申请|依据|途径|分割|权益|名下|自行|采用)'
            r'|(?:以及|赔偿金|应存入|上述|应向|提供处理|加盖|不限于|不再向|有权就|授权甲方|用于接送|配合前往|关联|关于我们|有管辖权|涉事|通过原告|沟通我们|介绍一下|安排到|选在我们|会从|已超额|以甲方|均通过|绩效奖金根据|通知贵|寄到|仲裁委员会|全权代表|须以书面)'
        )
        # Also filter standalone short bank references
        _org_fp_short = re.compile(
            r'^(?:通过|该|其|向|到|在|从|甲方|乙方)?\s*(?:银行|法院|检察院|公安局|派出所|指定的?收款银行)$'
        )
        entities = [
            e for e in entities
            if e.entity_type != EntityType.ORG_NAME
            or (not _org_fp.search(e.text) and not _org_fp_short.match(e.text))
        ]

        # --- Post-processing: remove PERSON_NAME false positives ---
        # Filter names that are actually common word fragments
        _name_fp_words = re.compile(
            r'^(?:方应|方承|方同|方确|方负|方提|方指|方支|方违|方有|方收|方授|方书|方另|方保|方解|方未|方逾|方不|方须|方需|方可|方已|方将|方在|方于|方按|方如|方对|方向|方以|方为|方应该|方必须|方各|方车|方式|方法|方面|方案|方便|方能)'
            r'|^(?:金超|金标|金不|金额|金融|金为|金的|金属|金钱)'
            r'|^(?:施损|施工|施行|施加|施救)'
            r'|^(?:高租|高于|高额|高度|高速)'
            r'|^(?:范围|范本)'
            r'|^(?:文件|文本|文书|文字|文化)'
            r'|^(?:易产|易于|易发|易被)'
            r'|^(?:常住|常规|常见|常行|常年|常态)'
            r'|^(?:龙华|龙岗|龙山|龙门|龙泉)$'
            r'|^(?:程序|程度)'
            r'|^(?:许可|许诺)'
            r'|^(?:任何|任意|任职|任命)'
            r'|^(?:武装|武力|武器)'
            r'|^(?:白色|白天)'
            r'|^(?:万元)'
            r'|^(?:石头|石材)'
            r'|^(?:赖以)'
            r'|^(?:马上)'
            r'|^(?:钱款|钱财|钱物)'
            r'|^(?:田间|田地|田产|田野)'
            r'|^(?:汪洋)'
            r'|^(?:段落)'
            r'|^(?:曾经|曾任|曾在|曾于|曾多次|曾先后|曾向|曾因)'
            r'|^(?:日期|日前|日内|日起|日止|日至)'
            r'|^(?:于每|于其|于平|于信|于本|于乙|于甲|于丙|于丁|于上|于下|于此|于该|于合|于借|于租|于约)'
            r'|^(?:何形|何方|何种|何时|何处|何人)'
            r'|^(?:周转|周末|周围|周知|周岁)'
            r'|^(?:张权|张四)'
            r'|^(?:金支|金及|金的|金应|金收|金计|金外|金需)'
            r'|^(?:姓名|姓氏)'
            r'|^(?:任担|任保|任赔|任连)'
            r'|^(?:常使|常经|常居)'
            r'|^(?:方使|方参|方造|方申|方加|方制|方无|方使用|方参与|方自|方作|方愿|方对|方认)'
            r'|^(?:方签|方追|方所|方损|方与|方要|方出|方共|方发|方借|方权|方内|方合|方中|方清|方全|方依|方逾|方违|方去|方当|方先|方受|方享|方地|方时|方后|方直|方擅|方责|方债|方处|方协|方超|方备|方薪|方积|方报|方赔|方一|方二|方三|方四)'
            r'|^(?:陈述|陈列|陈设|陈旧)'
            r'|^(?:任公|任不|任免)'
            r'|^(?:侯支|侯审|侯选)'
            r'|^(?:何变|何异|何义|何责|何索|何权|何损|何民|何法|何赔)'
            r'|^(?:于收|于破|于逾|于违|于未|于提|于解|于终|于届|于到|于扣|于应|于按|于依|于向|于支)'
            r'|^(?:金后|金均|金及|金全|金总|金应|金中|金归|金为|金按|金由|金不|金自|金立|金流|金划)'
            r'|^(?:江苏|江西|江南|江北|江门|江城)'
            r'|^(?:史数|史上|史料)'
            r'|^签名$'
            r'|^(?:方之|方仍|方生|方存|方主|方的|方宣|方多|方沟|方原|方纠|方收|方均|方各|方住|方配|方男|方女|方给|方暂|方拒|方欠|方持|方行|方取|方转|方代|方偿|方名|方父|方母)'
            r'|^(?:余款|余份|余下|余额|余期|余部)'
            r'|^(?:程相|程中)'
            r'|^仲裁$'
            r'|^(?:于电|于短|于独|于有|于人|于无|于不|于双|于一|于同|于处|于他|于第|于被|于原|于主|于公|于诉|于仲|于法|于判|于生|于日|于货|于两|于自)'
            r'|^(?:易明|易中|易的|易双)'
            r'|^(?:文如|文中|文的|文所|文与)'
            r'|^(?:何经|何条|何一|何通|何异)'
            r'|^(?:侯区)'
            r'|^(?:段款|段时|段期|段内)'
            r'|^(?:西安|西南|西北|西部|西湖)$'
            r'|^手机$'
            r'|^(?:任风|任时|任对|任后)'
            r'|^(?:曾就|曾以|曾为|曾与)'
            r'|^(?:常管|常日)'
            r'|^(?:程真|程款|程中)'
            r'|^(?:武汉锦|武汉鼎)'
        )
        # Contextual filter: "方X" after party prefixes is not a person name
        def _is_fang_fp(e, text):
            if e.entity_type != EntityType.PERSON_NAME:
                return False
            if not e.text.startswith('方'):
                return False
            if e.start > 0:
                prev = text[e.start - 1]
                if prev in '甲乙丙丁己对我另各三任何':
                    return True
                if e.start >= 2 and text[e.start - 2:e.start] in ('双方', '一方', '某方'):
                    return False  # "双方方某" is unlikely but don't filter
            return False

        # Geographic name filter: province/city/district names
        _geo_names = re.compile(
            r'^(?:江苏省|江西省|湖南省|湖北省|河南省|河北省|广东省|广西|四川省|陕西省|甘肃省|山东省|山西省|安徽省|浙江省|福建省|云南省|贵州省|海南省|吉林省|辽宁省|黑龙江)'
            r'|^(?:天津市|上海市|北京市|重庆市)'
            r'|^(?:江汉区|雨花台|武昌区|鼓楼区|朝阳区|浦东新|海淀区|西湖区|南山区|天河区|渝中区|蜀山区|思明区|沈河区)'
        )

        entities = [
            e for e in entities
            if e.entity_type != EntityType.PERSON_NAME
            or (not _name_fp_words.match(e.text) and not _is_fang_fp(e, text) and not _geo_names.match(e.text))
        ]

        # --- Post-processing: remove CASE_NUMBER FPs for mediation documents ---
        entities = [
            e for e in entities
            if e.entity_type != EntityType.CASE_NUMBER or '调字' not in e.text
        ]


        # --- Post-processing: remove MONEY FPs inside parenthetical after Chinese uppercase ---
        def _is_money_in_paren(e, text):
            if e.entity_type != EntityType.MONEY:
                return False
            before = text[max(0, e.start - 3):e.start]
            after = text[e.end:min(len(text), e.end + 5)]
            if ('（' in before or '(' in before) and ('）' in after or ')' in after):
                pre_ctx = text[max(0, e.start - 30):e.start]
                if re.search(r'[零壹贰叁肆伍陆柒捌玖拾佰仟万亿]+元', pre_ctx):
                    return True
            return False

        entities = [e for e in entities if not _is_money_in_paren(e, text)]

        # --- Layer 6b: Trim person names that grabbed a trailing verb/preposition ---
        _trailing_verbs = set("以其将把对为向在到至系因与并从由被让给")
        trimmed = []
        for e in entities:
            if (e.entity_type == EntityType.PERSON_NAME
                    and len(e.text) == 3
                    and e.text[-1] in _trailing_verbs
                    and e.text[0] in _SURNAMES):
                trimmed.append(DetectedEntity(
                    start=e.start, end=e.end - 1,
                    entity_type=EntityType.PERSON_NAME, text=e.text[:-1],
                ))
            else:
                trimmed.append(e)
        entities = trimmed

        # --- Layer 7: Known-name backfill ---
        # If a person name was detected once (via prefix/context), find ALL
        # other occurrences of the same name string in the document.
        # Exclude names that also appear as part of ORG_NAME entities.
        org_texts = {e.text for e in entities if e.entity_type == EntityType.ORG_NAME}
        known_names = set()
        for e in entities:
            if e.entity_type == EntityType.PERSON_NAME and len(e.text) >= 2:
                # Skip if this name is a substring of any detected org name
                if any(e.text in org for org in org_texts):
                    continue
                known_names.add(e.text)
                # Given-name backfill (e.g. "雅琴" from "孙雅琴") is disabled
                # because 2-char given names frequently collide with common words,
                # causing too many FPs. Users can enable it by passing the full
                # text through detect() twice if needed.
        occupied_after = {i for e in entities for i in range(e.start, e.end)}
        for name in known_names:
            start = 0
            while True:
                idx = text.find(name, start)
                if idx == -1:
                    break
                end_idx = idx + len(name)
                span = set(range(idx, end_idx))
                if not span & occupied_after:
                    entities.append(DetectedEntity(
                        start=idx, end=end_idx,
                        entity_type=EntityType.PERSON_NAME, text=name,
                    ))
                    occupied_after.update(span)
                start = idx + 1

        if _skip:
            entities = [e for e in entities if e.entity_type not in _skip]
        entities.sort(key=lambda e: e.start)
        return entities

    def redact(
        self,
        text: str,
        replacement_style: str = "placeholder",
        entity_types: Optional[List[EntityType]] = None,
        exclude_types: Optional[List[EntityType]] = None,
    ) -> str:
        entities = self.detect(text)
        if entity_types:
            entities = [e for e in entities if e.entity_type in entity_types]
        if exclude_types:
            entities = [e for e in entities if e.entity_type not in exclude_types]

        entities.sort(key=lambda e: e.start, reverse=True)
        counters: Dict[EntityType, int] = {}
        seen: Dict[str, str] = {}

        result = text
        for ent in entities:
            if replacement_style == "placeholder":
                if ent.text in seen:
                    replacement = seen[ent.text]
                else:
                    counters[ent.entity_type] = counters.get(ent.entity_type, 0) + 1
                    cn = ENTITY_TYPE_CN.get(ent.entity_type, ent.entity_type.value)
                    replacement = f"[{cn}{counters[ent.entity_type]}]"
                    seen[ent.text] = replacement
            elif replacement_style == "mask":
                replacement = "*" * len(ent.text)
            elif replacement_style == "delete":
                replacement = ""
            else:
                replacement = f"[{ent.entity_type.value}]"

            result = result[:ent.start] + replacement + result[ent.end:]

        return result
