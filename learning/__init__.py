from learning.tsp.decoding import decode as decode_tsp_like
from learning.pctsp.decoding import decode as decode_pctsp
from learning.cvrp.decoding import decode as decode_cvrp_like
from learning.op.decoding import decode as decode_op
from learning.cvrptw.decoding import decode as decode_cvrptw
from learning.mvc.decoding import decode as decode_mvc
from learning.mis.decoding import decode as decode_mis
from learning.kp.decoding import decode as decode_kp
from learning.umsp.decoding import decode as decode_upms
from learning.jssp.decoding import decode as decode_jssp_like
from learning.mclp.decoding import decode as decode_mclp


decoding_fn = { "tsp": decode_tsp_like, "trp": decode_tsp_like, "sop": decode_tsp_like,
                "cvrp": decode_cvrp_like, "sdcvrp": decode_cvrp_like, "ocvrp": decode_cvrp_like, "dcvrp": decode_cvrp_like,
                "cvrptw": decode_cvrptw,
                "op": decode_op,
                "pctsp": decode_pctsp,
                "kp": decode_kp,
                "mvc": decode_mvc,
                "mis": decode_mis,
                "mclp": decode_mclp,
                "upms": decode_upms,
                "jssp": decode_jssp_like, "ossp": decode_jssp_like}

