import cv2

# ==========================================
# CARREGA O CLASSIFICADOR HAAR CASCADE
# ==========================================

classificador = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ==========================================
# CARREGA A IMAGEM DO ÓCULOS
# ==========================================

Oculos = cv2.imread("Oculos.png", cv2.IMREAD_UNCHANGED)

# Verifica se o óculos foi carregado
if Oculos is None:
    print("Erro: não foi possível carregar o arquivo Oculos.png")
    exit()

# Verifica se possui transparência
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

    # Captura um frame da webcam
    ret, frame = camera.read()

    if not ret:
        break

    # ==========================================
    # CONVERTE PARA TONS DE CINZA
    # ==========================================

    cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # DETECTA OS ROSTOS
    # ==========================================

    rostos = classificador.detectMultiScale(
        cinza,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ==========================================
    # PERCORRE OS ROSTOS ENCONTRADOS
    # ==========================================

    for (x, y, largura, altura) in rostos:

        # ==========================================
        # RETÂNGULO DO TRACKING
        # ==========================================

        cv2.rectangle(
            frame,
            (x, y),
            (x + largura, y + altura),
            (0, 255, 0),
            2
        )

        # ==========================================
        # TEXTO "ROSTO DETECTADO"
        # ==========================================

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
        # TAMANHO DO ÓCULOS
        # ==========================================

        # O óculos fica um pouco maior que o rosto
        nova_largura = int(largura * 1.15)

        # Mantém a proporção original da imagem
        nova_altura = int(
            Oculos.shape[0] *
            (nova_largura / Oculos.shape[1])
        )

        # Redimensiona o óculos
        oculos_redimensionado = cv2.resize(
            Oculos,
            (nova_largura, nova_altura)
        )

        # ==========================================
        # POSIÇÃO DO ÓCULOS
        # ==========================================

        # Centraliza horizontalmente
        oculos_x = x - int((nova_largura - largura) / 2)

        # Posiciona na região dos olhos
        oculos_y = y + int(altura * 0.05)

        # ==========================================
        # VERIFICA OS LIMITES DA IMAGEM
        # ==========================================

        if oculos_x < 0:
            continue

        if oculos_y < 0:
            continue

        if oculos_x + nova_largura > frame.shape[1]:
            continue

        if oculos_y + nova_altura > frame.shape[0]:
            continue

        # ==========================================
        # TRANSPARÊNCIA DO PNG
        # ==========================================

        b, g, r, alpha = cv2.split(
            oculos_redimensionado
        )

        # Máscara da parte visível
        mascara = alpha

        # Máscara invertida
        mascara_invertida = cv2.bitwise_not(
            mascara
        )

        # ==========================================
        # REGIÃO ONDE O ÓCULOS SERÁ COLOCADO
        # ==========================================

        regiao = frame[
            oculos_y:
            oculos_y + nova_altura,

            oculos_x:
            oculos_x + nova_largura
        ]

        # ==========================================
        # REMOVE A PARTE TRANSPARENTE
        # ==========================================

        fundo = cv2.bitwise_and(
            regiao,
            regiao,
            mask=mascara_invertida
        )

        # ==========================================
        # CRIA A FRENTE DO ÓCULOS
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
        # COLOCA O ÓCULOS NA IMAGEM
        # ==========================================

        frame[
            oculos_y:
            oculos_y + nova_altura,

            oculos_x:
            oculos_x + nova_largura
        ] = resultado

    # ==========================================
    # MOSTRA A QUANTIDADE DE ROSTOS
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

    # ESC para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ==========================================
# ENCERRA
# ==========================================

camera.release()
cv2.destroyAllWindows()
