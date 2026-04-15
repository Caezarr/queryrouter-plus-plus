# QueryRouter++ pour LibreChat - Mode d'emploi

Guide utilisateur ultra-simple pour l'intégration LibreChat avec QueryRouter++.

## Les 4 modes disponibles

| Mode | Icône | Description | Quand l'utiliser ? |
|------|-------|-------------|-------------------|
| **Écologique** | 🌱 | Privilégie l'empreinte carbone | Vous souhaitez réduire votre impact environnemental |
| **Performance** | ⚡ | La meilleure qualité possible | La qualité de la réponse est prioritaire |
| **Économique** | 💰 | Le moins cher possible | Vous avez un budget serré |
| **Équilibré** | ⚖️ | Bon rapport qualité/prix | Usage quotidien (mode par défaut) |

## Comment choisir ?

```
Votre priorité ?
├── Qualité maximale → Mode Performance ⚡
├── Budget limité → Mode Économique 💰
├── Éco-responsable → Mode Écologique 🌱
└── Je ne sais pas → Mode Équilibré ⚖️
```

## Ce que vous voyez dans LibreChat

### Pendant la conversation
```
Modèle: ▼ Mode Équilibré

Utilisateur: Comment fonctionne le routing ?

Assistant: Le routing intelligent sélectionne...
```

### Après la réponse
```
Modèle: ▼ Mode Équilibré

Assistant: Voici la réponse...

— Généré par Gemini 2.5 Flash
```

## Configuration administrateur

Seul l'administrateur configure les modèles dans chaque mode.

### Éditer les modèles par mode

Fichier: `config/presets.yaml`

```yaml
presets:
  eco:
    allowed_models:
      - "gemini-2-5-flash"      # Modèles économes en énergie
      - "llama-4-maverick"
    
  performance:
    allowed_models:
      - "claude-opus-4-6"       # Modèles premium
      - "gpt-4-1"
      - "o3"
```

Redémarrer QueryRouter++ après modification.

## Installation rapide

### 1. Lancer QueryRouter++

```bash
# Docker
docker-compose -f docker-compose.4modes.yml up

# Ou natif
poetry run uvicorn queryrouter.api.main:app --port 8000
```

### 2. Configurer LibreChat

Copier `config/librechat-4modes.yaml` dans votre installation LibreChat:

```bash
cp config/librechat-4modes.yaml /chemin/vers/librechat/librechat.yaml
```

### 3. Définir les variables d'environnement

```bash
export QUERYROUTER_BASE_URL=http://localhost:8000
export QUERYROUTER_API_KEY=sk-queryrouter-local
export OPENAI_API_KEY=sk-...  # Au moins un provider requis
```

### 4. Redémarrer LibreChat

```bash
docker-compose restart  # ou npm run start
```

## FAQ

**Q: Puis-je voir les modèles disponibles dans un mode ?**  
R: Non, l'utilisateur ne voit que les 4 modes. Les modèles sont configurés par l'administrateur.

**Q: Puis-je changer de mode pendant une conversation ?**  
R: Oui, changez le modèle dans le dropdown LibreChat.

**Q: Comment savoir quel modèle a été utilisé ?**  
R: Le nom du modèle apparaît après chaque réponse (ex: "— Généré par Gemini 2.5 Flash").

**Q: Puis-je créer mes propres modes ?**  
R: Oui, en modifiant `config/presets.yaml` et en redémarrant QueryRouter++.

**Q: Quels providers sont supportés ?**  
R: OpenAI, Google (Gemini), DeepSeek, Meta (Together), Mistral, Alibaba.

## Dépannage

| Problème | Solution |
|----------|----------|
| "No model satisfies the given constraints" | Vérifiez que les clés API des providers sont configurées |
| Les modes n'apparaissent pas dans LibreChat | Vérifiez `QUERYROUTER_BASE_URL` et redémarrez LibreChat |
| Toujours le même modèle utilisé | Vérifiez que plusieurs modèles sont configurés dans le preset |

## Support

- Documentation: [README.md](../README.md)
- Issues: https://github.com/queryrouter/queryrouter-plus-plus/issues
