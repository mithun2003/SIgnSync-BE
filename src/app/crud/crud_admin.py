from fastcrud import FastCRUD

from ..models.sign_detection import SignDetection
from ..models.user import User

user_crud = FastCRUD(User)
detection_crud = FastCRUD(SignDetection)