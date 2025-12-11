# Frontend — FaceFiction (React + TypeScript)

Ce dossier contient l'application frontend de démonstration pour FaceFiction. L'application permet de soumettre des médias (images / frames) au backend FastAPI, d'afficher la prédiction (Real / Fake) et de visualiser des éléments d'interprétabilité (ex : heatmaps Grad-CAM).

## Introduction
Ce dossier contient le code source de l'interface utilisateur de la plateforme FaceFiction. L'interface permet aux utilisateurs de charger des médias (images / vidéos) et d'obtenir une prédiction d'authenticité (Real / Fake) avec des visualisations d'interprétabilité.

## Technologies Utilisées
- **Framework** : Next.js
- **Langage** : TypeScript
- **Styles** : CSS Modules + variables CSS globales
- **Gestion des dépendances** : npm

## Structure du Dossier (extrait)
```
frontend/
├── public/
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
├── next.config.ts
├── package.json
├── tsconfig.json
```

## Installation rapide
```bash
cd frontend
npm install
npm run dev
# ouvrir http://localhost:3000
```

## Contribution
1. Forkez le dépôt.
2. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/nom-fonctionnalite
   ```
3. Faites vos modifications et soumettez une pull request.

## Technologies Utilisées
- **Framework** : Next.js
- **Langage** : TypeScript
- **Styles** : CSS Modules
- **Gestion des dépendances** : npm

## Structure du Dossier
```
frontend/
├── public/
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   └── page.tsx
├── next.config.ts
├── package.json
├── tsconfig.json
```

## Installation
1. Accédez au dossier `frontend/` :
   ```bash
   cd frontend
   ```
2. Installez les dépendances :
   ```bash
   npm install
   ```

## Lancement du Serveur de Développement
1. Démarrez le serveur :
   ```bash
   npm run dev
   ```
2. Ouvrez votre navigateur et accédez à :
   ```
   http://localhost:3000
   ```

## Scripts Disponibles
- **`npm run dev`** : Démarre le serveur de développement.
- **`npm run build`** : Génère une version de production.
- **`npm start`** : Démarre le serveur en mode production.

## Contribution
1. Forkez le dépôt.
2. Créez une branche pour votre fonctionnalité :
   ```bash
   git checkout -b feature/nom-fonctionnalite
   ```
3. Faites vos modifications et soumettez une pull request.
