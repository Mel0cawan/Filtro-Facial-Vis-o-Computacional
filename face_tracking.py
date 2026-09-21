import cv2
import math

# ==========================================
# CARREGA OS CLASSIFICADORES
# ==========================================

classificador_rosto = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

classificador_olhos = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_eye.xml"
)

# ==========================================
# CARREGA A IMAGEM DO ÓCULOS
# ==========================================

Oculos = cv2.imread(
    "Oculos.png",
    cv2.IMREAD_UNCHANGED
)

if Oculos is None:
    print("Erro: não foi possível carregar o arquivo Oculos.png")
    exit()

if Oculos.shape[2] != 4:
    print("Erro: o Oculos.png precisa ter fundo transparente.")
    exit()

# ==========================================
# ACESSA A WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro: não foi possível acessar a webcam.")
    exit()

# ==========================================
# LOOP PRINCIPAL
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # ==========================================
    # CONVERTE PARA CINZA
    # ==========================================

    cinza = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    # ==========================================
    # DETECTA OS ROSTOS
    # ==========================================

    rostos = classificador_rosto.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ==========================================
    # PERCORRE OS ROSTOS
    # ==========================================

    for (x, y, largura, altura) in rostos:

        # ==========================================
        # DESENHA O RETÂNGULO DO ROSTO
        # ==========================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Rosto Detectado",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        # ==========================================
        # REGIÃO DOS OLHOS
        # ==========================================
        # Procura os olhos apenas na parte superior
        # do rosto para evitar barba, boca e queixo.

        altura_olhos = int(altura * 0.55)

        regiao_olhos = cinza[
            y:y + altura_olhos,
            x:x + largura
        ]

        # ==========================================
        # DETECTA OS OLHOS
        # ==========================================

        olhos = classificador_olhos.detectMultiScale(
            regiao_olhos,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(20, 20)
        )

        # ==========================================
        # PROCURA UM PAR DE OLHOS CONFIÁVEL
        # ==========================================

        par_olhos = None

        if len(olhos) >= 2:

            melhor_par = None
            menor_diferenca_y = float("inf")

            for i in range(len(olhos)):

                for j in range(i + 1, len(olhos)):

                    olho1 = olhos[i]
                    olho2 = olhos[j]

                    # Centro dos olhos dentro da região
                    olho1_x = olho1[0] + olho1[2] // 2
                    olho1_y = olho1[1] + olho1[3] // 2

                    olho2_x = olho2[0] + olho2[2] // 2
                    olho2_y = olho2[1] + olho2[3] // 2

                    # Diferença vertical entre os olhos
                    diferenca_y = abs(olho1_y - olho2_y)

                    # Distância horizontal
                    distancia_x = abs(olho1_x - olho2_x)

                    # Os olhos precisam estar aproximadamente
                    # na mesma altura.
                    mesma_altura = (
                        diferenca_y < altura * 0.12
                    )

                    # Eles precisam estar separados.
                    distancia_valida = (
                        distancia_x > largura * 0.15
                    )

                    if mesma_altura and distancia_valida:

                        if diferenca_y < menor_diferenca_y:

                            menor_diferenca_y = diferenca_y
                            melhor_par = (olho1, olho2)

            par_olhos = melhor_par

        # ==========================================
        # SÓ COLOCA O ÓCULOS SE ENCONTRAR
        # UM PAR DE OLHOS CONFIÁVEL
        # ==========================================

        if par_olhos is not None:

            olho1 = par_olhos[0]
            olho2 = par_olhos[1]

            # ==========================================
            # ORDENA OS OLHOS PELA POSIÇÃO X
            # ==========================================

            if olho1[0] > olho2[0]:
                olho1, olho2 = olho2, olho1

            # ==========================================
            # CENTRO DOS OLHOS
            # ==========================================

            olho1_x = olho1[0] + olho1[2] // 2
            olho1_y = olho1[1] + olho1[3] // 2

            olho2_x = olho2[0] + olho2[2] // 2
            olho2_y = olho2[1] + olho2[3] // 2

            # ==========================================
            # CONVERTE PARA COORDENADAS DA IMAGEM
            # ==========================================

            olho1_x_global = x + olho1_x
            olho1_y_global = y + olho1_y

            olho2_x_global = x + olho2_x
            olho2_y_global = y + olho2_y

            # ==========================================
            # DESENHA OS OLHOS DETECTADOS
            # ==========================================

            cv2.circle(
                frame,
                (olho1_x_global, olho1_y_global),
                5,
                (255, 0, 0),
                -1
            )

            cv2.circle(
                frame,
                (olho2_x_global, olho2_y_global),
                5,
                (255, 0, 0),
                -1
            )

            # ==========================================
            # CENTRO ENTRE OS DOIS OLHOS
            # ==========================================

            centro_x = int(
                (olho1_x_global + olho2_x_global) / 2
            )

            centro_y = int(
                (olho1_y_global + olho2_y_global) / 2
            )

            # ==========================================
            # DISTÂNCIA ENTRE OS OLHOS
            # ==========================================

            distancia = math.sqrt(
                (olho2_x_global - olho1_x_global) ** 2 +
                (olho2_y_global - olho1_y_global) ** 2
            )

            # ==========================================
            # TAMANHO DO ÓCULOS
            # ==========================================

            nova_largura = int(
                distancia * 2.4
            )

            nova_altura = int(
                Oculos.shape[0] *
                (nova_largura / Oculos.shape[1])
            )

            # ==========================================
            # EVITA TAMANHO INVÁLIDO
            # ==========================================

            if nova_largura <= 0 or nova_altura <= 0:
                continue

            # ==========================================
            # REDIMENSIONA O ÓCULOS
            # ==========================================

            oculos_redimensionado = cv2.resize(
                Oculos,
                (nova_largura, nova_altura)
            )

            # ==========================================
            # CALCULA A INCLINAÇÃO DA CABEÇA
            # ==========================================

            angulo = math.degrees(
                math.atan2(
                    olho2_y_global - olho1_y_global,
                    olho2_x_global - olho1_x_global
                )
            )

            # ==========================================
            # GIRA O ÓCULOS
            # ==========================================

            centro_oculos = (
                nova_largura // 2,
                nova_altura // 2
            )

            matriz_rotacao = cv2.getRotationMatrix2D(
                centro_oculos,
                angulo,
                1.0
            )

            oculos_rotacionado = cv2.warpAffine(
                oculos_redimensionado,
                matriz_rotacao,
                (nova_largura, nova_altura),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0, 0)
            )

            # ==========================================
            # POSIÇÃO DO ÓCULOS
            # ==========================================

            oculos_x = int(
                centro_x - nova_largura / 2
            )

            oculos_y = int(
                centro_y - nova_altura * 0.50
            )

            # ==========================================
            # VERIFICA LIMITES
            # ==========================================

            if oculos_x < 0:
                continue

            if oculos_y < 0:
                continue

            if (
                oculos_x + nova_largura
                > frame.shape[1]
            ):
                continue

            if (
                oculos_y + nova_altura
                > frame.shape[0]
            ):
                continue

            # ==========================================
            # TRANSPARÊNCIA
            # ==========================================

            b, g, r, alpha = cv2.split(
                oculos_rotacionado
            )

            mascara = alpha

            mascara_invertida = cv2.bitwise_not(
                mascara
            )

            # ==========================================
            # REGIÃO DA WEBCAM
            # ==========================================

            regiao = frame[
                oculos_y:
                oculos_y + nova_altura,

                oculos_x:
                oculos_x + nova_largura
            ]

            # ==========================================
            # FUNDO
            # ==========================================

            fundo = cv2.bitwise_and(
                regiao,
                regiao,
                mask=mascara_invertida
            )

            # ==========================================
            # FRENTE
            # ==========================================

            frente = cv2.merge(
                (b, g, r)
            )

            frente = cv2.bitwise_and(
                frente,
                frente,
                mask=mascara
            )

            # ==========================================
            # JUNTA O ÓCULOS COM A WEBCAM
            # ==========================================

            resultado = cv2.add(
                fundo,
                frente
            )

            # ==========================================
            # COLOCA O ÓCULOS NO FRAME
            # ==========================================

            frame[
                oculos_y:
                oculos_y + nova_altura,

                oculos_x:
                oculos_x + nova_largura
            ] = resultado

    # ==========================================
    # QUANTIDADE DE ROSTOS
    # ==========================================

    cv2.putText(
        frame,
        f"Rostos: {len(rostos)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # ==========================================
    # MOSTRA A WEBCAM
    # ==========================================

    cv2.imshow(
        "Filtro Facial",
        frame
    )

    # ==========================================
    # ESC PARA SAIR
    # ==========================================

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==========================================
# ENCERRA
# ==========================================

camera.release()
cv2.destroyAllWindows()