import logging
from flask import Flask, render_template, request, redirect, url_for, Response, flash, session, make_response, jsonify
from flask_socketio import SocketIO, emit
import mysql.connector
from mysql.connector import Error
import cv2
import numpy as np
import base64
import io
from PIL import Image
import threading
from datetime import datetime, timedelta
from fpdf import FPDF

# --- CONFIGURACIÓN PRINCIPAL ---
app = Flask(__name__)
app.secret_key = 'CLAVE_ULTRA_SECRETA_V5_SOCKETS'
socketio = SocketIO(app, cors_allowed_origins="*")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

ADMIN_PASSWORD = "admin"

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'facial_db'
}

CAMERA_INDEX = 0  # Cámara integrada de la ROG Strix
camera = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
reconocedor = cv2.face.LBPHFaceRecognizer_create()
modelo_entrenado = False
nombres_usuarios = {}
lock = threading.Lock()
ultimo_access_cache = {}

# --- BASE DE DATOS ---
def obtener_conexion():
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected(): return conn
    except Error as e:
        logging.error(f"Error BD: {e}")
        return None

def inicializar_tabla_muestras():
    conn = obtener_conexion()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rostro_muestras (
                id INT AUTO_INCREMENT PRIMARY KEY,
                usuario_id INT,
                imagen LONGBLOB,
                FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
            )
        """)
        conn.commit()
        conn.close()

inicializar_tabla_muestras()

# --- CONTROL DE CÁMARA ---
def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return camera

def liberar_camara():
    global camera
    if camera and camera.isOpened():
        camera.release()
        camera = None

# --- RECONOCIMIENTO FACIAL IA ---
def entrenar_modelo():
    global modelo_entrenado, nombres_usuarios
    conn = obtener_conexion()
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT id, nombre, es_admin FROM usuarios")
        usuarios_db = cursor.fetchall()
        nombres_usuarios.clear()
        for id_u, nom, es_admin in usuarios_db:
            nombres_usuarios[id_u] = {'nombre': nom, 'es_admin': es_admin}

        cursor.execute("SELECT usuario_id, imagen FROM rostro_muestras")
        muestras_db = cursor.fetchall()
        
        rostros, ids = [], []
        for id_u, img_blob in muestras_db:
            if img_blob:
                try:
                    img_np = np.array(Image.open(io.BytesIO(img_blob)).convert('L'), 'uint8')
                    faces = face_cascade.detectMultiScale(img_np, 1.1, 5)
                    for (x, y, w, h) in faces:
                        rostros.append(img_np[y:y+h, x:x+w])
                        ids.append(id_u)
                except: pass

        if rostros:
            reconocedor.train(rostros, np.array(ids))
            modelo_entrenado = True
            logging.info(f"Modelo IA entrenado con {len(rostros)} muestras de rostros.")
            return True
        modelo_entrenado = False
        return False
    finally:
        if conn and conn.is_connected(): conn.close()

def registrar_historial(id_usuario):
    global ultimo_access_cache
    now = datetime.now()
    if id_usuario in ultimo_access_cache and now - ultimo_access_cache[id_usuario] < timedelta(seconds=60):
        return False
    
    conn = obtener_conexion()
    if conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO accesos (usuario_id) VALUES (%s)", (id_usuario,))
        conn.commit()
        
        cursor.execute("SELECT u.nombre, DATE_FORMAT(a.fecha_acceso, '%H:%i:%s') as hora, u.es_admin FROM accesos a JOIN usuarios u ON a.usuario_id = u.id WHERE a.usuario_id = %s ORDER BY a.fecha_acceso DESC LIMIT 1", (id_usuario,))
        res = cursor.fetchone()
        conn.close()
        
        ultimo_access_cache[id_usuario] = now
        
        if res:
            nombre, hora, es_admin = res
            socketio.emit('nueva_asistencia', {'nombre': nombre, 'hora': hora, 'es_admin': es_admin})
            socketio.emit('login_exitoso', {'id': id_usuario, 'nombre': nombre, 'es_admin': es_admin, 'hora': hora})
        return True
    return False

def generar_frames(modo='reconocimiento'):
    cam = get_camera()
    while True:
        with lock:
            success, frame = cam.read()
        if not success: break

        if modo in ['reconocimiento', 'login']:
            if not modelo_entrenado: entrenar_modelo()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)

            for (x, y, w, h) in faces:
                roi = gray[y:y+h, x:x+w]
                id_p, conf = reconocedor.predict(roi) if modelo_entrenado else (-1, 999)
                
                UMBRAL = 55.0  # Umbral estricto para evitar confusiones

                if conf < UMBRAL:
                    datos_usuario = nombres_usuarios.get(id_p)
                    if datos_usuario:
                        nombre = datos_usuario['nombre']
                        es_admin = datos_usuario['es_admin']
                        registrar_historial(id_p)
                        color = (235, 100, 52) if es_admin else (46, 204, 113)
                    else:
                        nombre = "Desconocido"
                        color = (52, 73, 94)
                else:
                    nombre = "Desconocido"
                    color = (52, 73, 94)

                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, nombre, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- RUTAS DE LA APLICACIÓN ---
@app.before_request
def requerir_login():
    rutas_admin = ['dashboard', 'crear', 'editar', 'eliminar', 'guardar_usuario_multishot', 'exportar_pdf']
    if request.endpoint in rutas_admin and not session.get('admin_logged_in'):
        flash("🔒 Acceso restringido.", "warning")
        return redirect(url_for('login'))

@app.route('/')
def login():
    if session.get('admin_logged_in'): return redirect(url_for('dashboard'))
    return render_template('login.html')

# RUTA OPTIMIZADA CON RESPUESTA JSON RÁPIDA SIN BLOQUEOS
@app.route('/do_login_password', methods=['POST'])
def do_login_password():
    try:
        data = request.get_json(silent=True) or request.form
        password_ingresada = (data.get('password') or '').strip()
        
        if password_ingresada == ADMIN_PASSWORD:
            session['admin_logged_in'] = True
            session['admin_nombre'] = "Administrador Master"
            return jsonify({'status': 'ok', 'redirect': url_for('dashboard')})
        else:
            return jsonify({'status': 'error', 'message': 'Contraseña incorrecta.'}), 401
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/dashboard')
def dashboard():
    conn = obtener_conexion()
    usuarios, historial = [], []
    stats = {'total_usuarios': 0, 'asistencias_hoy': 0, 'admins': 0, 'modelo_listo': modelo_entrenado}
    if conn:
        c = conn.cursor(dictionary=True)
        c.execute("SELECT * FROM usuarios ORDER BY fecha_registro DESC")
        usuarios = c.fetchall()
        stats['total_usuarios'] = len(usuarios)
        stats['admins'] = sum(1 for u in usuarios if u.get('es_admin') == 1)

        for u in usuarios:
            u['imagen_src'] = None
            if u.get('fecha_registro'): 
                u['fecha_fmt'] = u['fecha_registro'].strftime("%d/%m/%Y")
        
        c.execute("SELECT a.fecha_acceso, u.nombre, u.es_admin FROM accesos a JOIN usuarios u ON a.usuario_id = u.id ORDER BY a.fecha_acceso DESC LIMIT 10")
        historial = c.fetchall()
        for h in historial: 
            h['fecha_fmt'] = h['fecha_acceso'].strftime("%d/%m/%Y %H:%M:%S")

        c.execute("SELECT COUNT(*) as hoy FROM accesos WHERE DATE(fecha_acceso) = CURDATE()")
        stats['asistencias_hoy'] = c.fetchone()['hoy']
        conn.close()
    return render_template('index.html', usuarios=usuarios, historial=historial, stats=stats)

@app.route('/exportar_pdf')
def exportar_pdf():
    conn = obtener_conexion()
    c = conn.cursor()
    c.execute("SELECT u.nombre, a.fecha_acceso FROM accesos a JOIN usuarios u ON a.usuario_id = u.id ORDER BY a.fecha_acceso DESC LIMIT 100")
    registros = c.fetchall()
    conn.close()

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(30, 41, 59)
            self.rect(0, 0, 210, 25, 'F')
            self.set_font('Arial', 'B', 16)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, 'BioSystem Pro - Reporte de Asistencias', 0, 1, 'C')
            self.ln(10)
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Pagina {self.page_no()}', 0, 0, 'C')

    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 11)
    pdf.set_fill_color(241, 245, 249)
    pdf.cell(100, 10, txt=" Usuario", border=1, fill=True)
    pdf.cell(90, 10, txt=" Fecha y Hora", border=1, ln=1, fill=True)
    pdf.set_font("Arial", size=10)
    for nombre, fecha in registros:
        pdf.cell(100, 9, txt=f" {str(nombre)}", border=1)
        pdf.cell(90, 9, txt=f" {str(fecha)}", border=1, ln=1)

    response = make_response(pdf.output(dest='S').encode('latin-1'))
    response.headers.set('Content-Disposition', 'attachment', filename='Reporte_Asistencias.pdf')
    response.headers.set('Content-Type', 'application/pdf')
    return response

@app.route('/crear')
def crear(): return render_template('crear.html')

@app.route('/reconocer')
def reconocer(): return render_template('reconocer.html')

@app.route('/video_feed/<modo>')
def video_feed(modo): return Response(generar_frames(modo), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/guardar_usuario_multishot', methods=['POST'])
def guardar_usuario_multishot():
    try:
        data = request.get_json()
        nombre = data.get('nombre')
        es_admin = 1 if data.get('es_admin') else 0
        fotos_b64 = data.get('fotos', [])

        if nombre and fotos_b64:
            conn = obtener_conexion()
            if conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO usuarios (nombre, es_admin) VALUES (%s, %s)", (nombre, es_admin))
                usuario_id = cursor.lastrowid

                for foto_b64 in fotos_b64:
                    img_bin = base64.b64decode(foto_b64.split(",", 1)[1])
                    cursor.execute("INSERT INTO rostro_muestras (usuario_id, imagen) VALUES (%s, %s)", (usuario_id, img_bin))
                
                conn.commit()
                conn.close()
                
                global modelo_entrenado
                modelo_entrenado = False
                return jsonify({'status': 'success', 'message': 'Usuario registrado exitosamente.'})
        return jsonify({'status': 'error', 'message': 'Datos incompletos.'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/eliminar/<int:id_u>')
def eliminar(id_u):
    conn = obtener_conexion()
    if conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM rostro_muestras WHERE usuario_id = %s", (id_u,))
        cursor.execute("DELETE FROM usuarios WHERE id = %s", (id_u,))
        conn.commit()
        conn.close()
        global modelo_entrenado
        modelo_entrenado = False
        flash("🗑️ Usuario eliminado.", "info")
    return redirect(url_for('dashboard'))

@app.route('/api/ultimas_asistencias')
def ultimas_asistencias():
    conn = obtener_conexion()
    if not conn: return jsonify([])
    try:
        c = conn.cursor(dictionary=True)
        c.execute("""
            SELECT u.nombre, DATE_FORMAT(a.fecha_acceso, '%H:%i:%s') as hora 
            FROM accesos a 
            JOIN usuarios u ON a.usuario_id = u.id 
            WHERE DATE(a.fecha_acceso) = CURDATE() 
            ORDER BY a.fecha_acceso DESC LIMIT 8
        """)
        return jsonify(c.fetchall())
    except: return jsonify([])
    finally: conn.close()

if __name__ == '__main__':
    try:
        socketio.run(app, debug=True, port=5001, host='0.0.0.0')
    finally:
        liberar_camara()