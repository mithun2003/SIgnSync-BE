from fastcrud import FastCRUD

from ..models.signs import Signs
from ..schemas.signs import SignCreate, SignRead, SignUpdate

CRUDSigns = FastCRUD[Signs, SignCreate, SignUpdate, None, None, SignRead]

crud_signs = CRUDSigns(Signs)
