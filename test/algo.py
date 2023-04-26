from hmacl.algo.algo import Algo
from hmacl.config import parser
from hmacl.utils.logging_ import get_logger


args = parser.parse_args()
algo = Algo(args, get_logger())
algo.run()


