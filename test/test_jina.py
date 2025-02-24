from llm.chroma import chroma

def test_get_similarity_whith_scores():
    result = chroma.get_similarity_whith_scores("cuál es el horario de atención de la pastelería")
    print(result)
    assert result[0][1] == 0.5265586972236633
    assert result[1][1] == 0.5246440768241882
    assert result[2][1] == 0.4540383815765381

def test_get_similarity():
    result = chroma.get_similarity("cuál es el horario de atención de la pastelería")
    print(result)
    data = result.split('\n')
    assert data[0] == "nuestro horario de atención en sucursal es de lunes a sábado de 09:00 a 18:00 hrs. y nuestros despachos a domicilio desde las 12:00 hrs hasta las 20:00 hrs"
    assert data[1] == "el telefono de la pastelería la palmera es 55 2 268988."
    # assert data[3] == """para situaciones no contempladas en estas políticas:
    # - contactar directamente a través del formulario de contacto
    # - cada caso será evaluado individualmente por pastelería la palmera"""