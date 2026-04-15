# QueryRouter++ × LibreChat - Guide d'Intégration Complète

Guide complet pour intégrer QueryRouter++ avec LibreChat en mode "4 Modes" ultra-simple.

---

## Table des matières

1. [Vue d'ensemble](#vue-densemble)
2. [Prérequis](#prérequis)
3. [Installation rapide](#installation-rapide)
4. [Configuration détaillée](#configuration-détaillée)
5. [Utilisation](#utilisation)
6. [Configuration avancée](#configuration-avancée)
7. [Dépannage](#dépannage)
8. [Architecture](#architecture)

---

## Vue d'ensemble

QueryRouter++ s'intègre à LibreChat via un **endpoint OpenAI-compatible**. L'utilisateur final ne voit que **4 modes simples** :

| Mode | Icône | Description | Cas d'usage |
|------|-------|-------------|-------------|
| **Écologique** | 🌱 | Privilégie l'empreinte carbone | Environnement, responsabilité |
| **Performance** | ⚡ | La meilleure qualité possible | Qualité critique, production |
| **Économique** | 💰 | Le moins cher possible | Budget serré, tests |
| **Équilibré** | ⚖️ | Bon rapport qualité/prix | Usage quotidien (défaut) |

**Ce que voit l'utilisateur :**
```
Modèle: ▼ Mode Équilibré

Utilisateur: Explique le machine learning

Assistant: Le machine learning est...

— Généré par Gemini 2.5 Flash
```

**Ce que configure l'admin :**
- Les modèles disponibles dans chaque mode (`config/presets.yaml`)
- Les poids (performance, coût, latence, écologie)
- La stratégie de routage (direct, cascade)

---

## Prérequis

### Système
- Docker et Docker Compose
- Au moins 2GB RAM disponible
- Ports 8000 (QueryRouter) et 3080 (LibreChat) disponibles

### Comptes API (au moins un provider)
Vous avez besoin d'au moins une clé API parmi :
- **OpenAI** (GPT-4, GPT-4o, o3...) → [platform.openai.com](https://platform.openai.com)
- **Google** (Gemini) → [ai.google.dev](https://ai.google.dev)
- **DeepSeek** (modèles économiques) → [deepseek.com](https://deepseek.com)
- **Anthropic** (Claude) → [anthropic.com](https://anthropic.com)
- **Mistral** → [mistral.ai](https://mistral.ai)
- **Together AI** (Meta LLaMA) → [together.ai](https://together.ai)
- **Alibaba Cloud** (Qwen) → [aliyun.com](https://aliyun.com)

---

## Installation rapide (5 minutes)

### 1. Cloner et configurer

```bash
# Cloner le repository
git clone https://github.com/Caezarr/queryrouter-plus-plus.git
cd queryrouter-plus-plus

# Copier le template de configuration
cp .env.4modes.example .env

# Éditer le fichier .env avec vos clés API
nano .env  # ou code .env, vim .env, etc.
```

**Exemple de fichier `.env` :**
```bash
# QueryRouter++
QUERYROUTER_PORT=8000
QUERYROUTER_API_KEY=sk-queryrouter-local

# Providers (au moins un requis)
OPENAI_API_KEY=sk-votre-clé-openai
GOOGLE_API_KEY=votre-clé-google
DEEPSEEK_API_KEY=votre-clé-deepseek

# MongoDB (pour LibreChat)
MONGO_ROOT_USERNAME=admin
MONGO_ROOT_PASSWORD=password123

# LibreChat
LIBRECHAT_PORT=3080
```

### 2. Lancer les services

```bash
# Lancer QueryRouter++ seul (sans LibreChat)
docker-compose -f docker-compose.4modes.yml up -d queryrouter

# OU lancer la stack complète (QueryRouter++ + LibreChat)
docker-compose -f docker-compose.4modes.yml --profile with-librechat up -d
```

### 3. Vérifier l'installation

```bash
# Health check QueryRouter++
curl http://localhost:8000/health
# Attendu: {"status": "ok", "version": "0.2.0"}

# Liste des modes disponibles
curl http://localhost:8000/v1/models
# Attendu: [mode-ecologique, mode-performance, mode-economique, mode-equilibre]
```

### 4. Accéder à l'interface

- **QueryRouter++ API** : http://localhost:8000/docs (Swagger UI)
- **LibreChat** : http://localhost:3080 (si activé)

---

## Configuration détaillée

### Étape 1 : Configurer les modèles par mode

Éditer `config/presets.yaml` pour définir quels modèles sont disponibles dans chaque mode :

```yaml
presets:
  eco:
    name: "Mode Écologique"
    description: "Privilégie l'empreinte carbone"
    icon: "🌱"
    allowed_models:
      - "gemini-2-5-flash"      # 15g CO2/MTok
      - "llama-4-maverick"      # 18g CO2/MTok
      - "qwen-3-235b"           # 22g CO2/MTok
    weights:
      w_performance: 0.15
      w_cost: 0.10
      w_latency: 0.10
      w_ecology: 0.65
    strategy: "direct"

  performance:
    name: "Mode Performance"
    description: "La meilleure qualité possible"
    icon: "⚡"
    allowed_models:
      - "claude-opus-4-6"       # MMLU: 91.1%
      - "gpt-4-1"               # MMLU: 90.2%
      - "o3"                    # MMLU: 92.9%
      - "gemini-2-5-pro"        # MMLU: 89.8%
    weights:
      w_performance: 0.85
      w_cost: 0.05
      w_latency: 0.05
      w_ecology: 0.05
    strategy: "direct"

  economique:
    name: "Mode Économique"
    description: "Le moins cher possible"
    icon: "💰"
    allowed_models:
      - "deepseek-v3"          # $0.42/MTok output
      - "qwen-3-235b"          # $1.50/MTok output
      - "gemini-2-5-flash"     # $2.50/MTok output
    weights:
      w_performance: 0.10
      w_cost: 0.80
      w_latency: 0.05
      w_ecology: 0.05
    strategy: "cascade"
    cascade_threshold: 0.6

  equilibre:
    name: "Mode Équilibré"
    description: "Bon rapport qualité/prix"
    icon: "⚖️"
    allowed_models: null  # Tous les modèles disponibles
    weights:
      w_performance: 0.30
      w_cost: 0.40
      w_latency: 0.15
      w_ecology: 0.15
    strategy: "direct"
```

**Redémarrer après modification :**
```bash
docker-compose -f docker-compose.4modes.yml restart queryrouter
```

### Étape 2 : Intégrer à LibreChat existant

Si vous avez déjà une instance LibreChat, ajoutez ce bloc à votre `librechat.yaml` :

```yaml
endpoints:
  custom:
    - name: "QueryRouter++"
      apiKey: "${QUERYROUTER_API_KEY}"
      baseURL: "${QUERYROUTER_BASE_URL}/v1"
      
      # Les 4 modes virtuels
      models:
        default:
          - "mode-ecologique"
          - "mode-performance"
          - "mode-economique"
          - "mode-equilibre"
        fetch: false
      
      modelDisplayLabel: "Mode"
      titleConvo: true
      titleModel: "mode-equilibre"
      
      dropParams:
        - "user"
        - "frequency_penalty"
        - "presence_penalty"
```

**Variables d'environnement à ajouter :**
```bash
export QUERYROUTER_BASE_URL=http://localhost:8000
export QUERYROUTER_API_KEY=sk-queryrouter-local
```

Redémarrer LibreChat :
```bash
docker-compose restart librechat  # ou npm run start si natif
```

---

## Utilisation

### Pour l'utilisateur final

1. **Ouvrir LibreChat** : http://localhost:3080
2. **Sélectionner un mode** dans le dropdown "Modèle" :
   - 🌱 Mode Écologique
   - ⚡ Mode Performance
   - 💰 Mode Économique
   - ⚖️ Mode Équilibré (défaut)

3. **Converser normalement**

4. **Voir le modèle utilisé** : Après chaque réponse, LibreChat affiche :
   ```
   — Généré par [Nom du Modèle]
   ```

### Exemples de conversations

**Mode Écologique :**
```
Utilisateur: Résume cet article sur le climat
Assistant: [Réponse] 
— Généré par Gemini 2.5 Flash (empreinte: 15g CO2)
```

**Mode Performance :**
```
Utilisateur: Débug ce code Python complexe
Assistant: [Réponse détaillée]
— Généré par Claude Opus 4.6
```

**Mode Économique :**
```
Utilisateur: Traduis ce texte en espagnol
Assistant: [Traduction]
— Généré par DeepSeek V3 ($0.0004)
```

---

## Configuration avancée

### Personnaliser les poids

Dans `config/presets.yaml`, les poids définissent l'importance relative :

```yaml
weights:
  w_performance: 0.40  # 40% - importance de la qualité
  w_cost: 0.40        # 40% - importance du prix
  w_latency: 0.10     # 10% - importance de la vitesse
  w_ecology: 0.10     # 10% - importance écologique
```

**Doivent toujours sommer à 1.0**

### Changer la stratégie

```yaml
strategy: "direct"    # Score tous les modèles, choisit le meilleur
# OU
strategy: "cascade"   # Essaye les moins chers d'abord
```

### Ajouter un nouveau mode

1. Créer une entrée dans `config/presets.yaml` :
```yaml
presets:
  mon_mode:
    name: "Mon Mode Perso"
    description: "Description"
    icon: "🔧"
    allowed_models:
      - "gpt-4-1-mini"
      - "gemini-2-5-flash"
    weights:
      w_performance: 0.5
      w_cost: 0.5
      w_latency: 0.0
      w_ecology: 0.0
    strategy: "direct"
```

2. Ajouter à `config/librechat-4modes.yaml` :
```yaml
models:
  default:
    - "mode-ecologique"
    - "mode-performance"
    - "mode-economique"
    - "mode-equilibre"
    - "mode-mon_mode"  # ← Ajouter ici
```

3. Redémarrer

### Modèles disponibles

Voir la liste complète des modèles supportés :
```bash
curl http://localhost:8000/models
```

---

## Dépannage

### Problèmes courants

#### "No model satisfies the given constraints"

**Cause :** Aucun modèle du preset n'a de clé API configurée.

**Solution :**
```bash
# Vérifier les variables d'environnement
docker-compose -f docker-compose.4modes.yml exec queryrouter env | grep API_KEY

# Vérifier que au moins un provider du preset est configuré
# Exemple: si eco utilise gemini et deepseek, vérifier:
# GOOGLE_API_KEY et DEEPSEEK_API_KEY sont définis
```

#### Les modes n'apparaissent pas dans LibreChat

**Cause :** Configuration LibreChat incorrecte ou cache.

**Solution :**
```bash
# Vérifier la config
cat config/librechat-4modes.yaml | grep -A5 "models:"

# Redémarrer LibreChat
docker-compose -f docker-compose.4modes.yml restart librechat

# Vider le cache navigateur (Ctrl+Shift+R)
```

#### Toujours le même modèle utilisé

**Cause :** Un seul modèle configuré dans le preset, ou un modèle domine toujours.

**Solution :**
```bash
# Vérifier le preset
cat config/presets.yaml | grep -A10 "eco:"

# Vérifier que plusieurs modèles sont listés dans allowed_models
```

#### Erreur de connexion à QueryRouter++

**Cause :** QueryRouter++ n'est pas accessible.

**Solution :**
```bash
# Vérifier que QueryRouter++ tourne
docker-compose -f docker-compose.4modes.yml ps

# Vérifier les logs
docker-compose -f docker-compose.4modes.yml logs queryrouter

# Tester l'endpoint
curl http://localhost:8000/health
```

### Logs et debugging

```bash
# Logs QueryRouter++
docker-compose -f docker-compose.4modes.yml logs -f queryrouter

# Logs LibreChat
docker-compose -f docker-compose.4modes.yml logs -f librechat

# Tous les logs
docker-compose -f docker-compose.4modes.yml logs -f
```

### Reset complet

```bash
# Arrêter tout
docker-compose -f docker-compose.4modes.yml down

# Supprimer les volumes (⚠️ perd les données)
docker-compose -f docker-compose.4modes.yml down -v

# Reconstruire
docker-compose -f docker-compose.4modes.yml up -d --build
```

---

## Architecture

### Flux de données

```
Utilisateur (LibreChat)
    ↓
Sélectionne "Mode Écologique"
    ↓
POST /v1/chat/completions
model: "mode-ecologique"
    ↓
QueryRouter++
    ↓
Résout mode-ecologique → preset "eco"
    ↓
Charge allowed_models: [flash, maverick, qwen]
    ↓
Score les modèles avec poids écologie=0.65
    ↓
Sélectionne gemini-2-5-flash (meilleur score éco)
    ↓
Proxy vers Google Gemini API
    ↓
Retourne réponse + metadata
    ↓
LibreChat affiche réponse + "— Généré par Gemini 2.5 Flash"
```

### Fichiers importants

| Fichier | Rôle |
|---------|------|
| `config/presets.yaml` | Configuration des 4 modes (modèles, poids, stratégies) |
| `config/librechat-4modes.yaml` | Configuration LibreChat |
| `api/presets.py` | Module de chargement des presets |
| `api/openai_compat.py` | Endpoint OpenAI-compatible avec mapping modes→presets |
| `docker-compose.4modes.yml` | Stack Docker complète |

---

## Référence API

### Endpoints QueryRouter++

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/models` | Liste les 4 modes disponibles |
| `POST /v1/chat/completions` | Chat avec routage intelligent |

### Format de réponse (metadata)

```json
{
  "choices": [...],
  "queryrouter": {
    "mode": "eco",
    "mode_name": "Mode Écologique",
    "mode_icon": "🌱",
    "model_used": "gemini-2-5-flash",
    "model_display_name": "Gemini 2.5 Flash",
    "provider": "Google",
    "preference": "ecology",
    "explanation": "Direct routing selected gemini-2-5-flash with score 0.823",
    "scores": [
      {"model_id": "gemini-2-5-flash", "score": 0.823},
      {"model_id": "llama-4-maverick", "score": 0.791}
    ]
  }
}
```

---

## Support

- **Documentation** : [docs/LIBRECHAT_4MODES.md](LIBRECHAT_4MODES.md)
- **Issues** : https://github.com/Caezarr/queryrouter-plus-plus/issues
- **Discussion LibreChat** : https://github.com/danny-avila/LibreChat/discussions

---

## Licence

MIT License - Copyright (c) 2026 QueryRouter++ Team
