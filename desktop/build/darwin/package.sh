#!/usr/bin/env bash
# Build Pipeline.app and a DMG the in-app updater recognises.
#   build/darwin/package.sh 0.0.14          -> bin/pipeline-0.0.14-darwin-arm64.dmg + bin/SHA256SUMS
# Run from desktop/. Ad-hoc signed: no Developer ID, so Gatekeeper asks on first open.
set -euo pipefail

VERSION="${1:?usage: package.sh <version, e.g. 0.0.14>}"
VERSION="${VERSION#v}"
ARCH="$(go env GOARCH)"
APP="bin/Pipeline.app"
DMG="bin/pipeline-${VERSION}-darwin-${ARCH}.dmg"

# The React app (vite build -> frontend/dist/index.html + static/) plus the legacy page at dist/legacy.html.
[[ -d frontend/node_modules ]] || npm --prefix frontend ci
npm --prefix frontend run build
node frontend/scripts/copy-factory.mjs

mkdir -p bin
CGO_ENABLED=1 MACOSX_DEPLOYMENT_TARGET=14.0 CGO_CFLAGS="-mmacosx-version-min=14.0" CGO_LDFLAGS="-mmacosx-version-min=14.0" \
  go build -tags production -trimpath -buildvcs=false -ldflags "-w -s -X main.version=${VERSION}" -o bin/pipeline

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp bin/pipeline "$APP/Contents/MacOS/pipeline"
cp build/darwin/icons.icns "$APP/Contents/Resources/"
[[ -f build/darwin/Assets.car ]] && cp build/darwin/Assets.car "$APP/Contents/Resources/"
cp build/darwin/Info.plist "$APP/Contents/Info.plist"
plutil -replace CFBundleVersion -string "$VERSION" "$APP/Contents/Info.plist"
plutil -replace CFBundleShortVersionString -string "$VERSION" "$APP/Contents/Info.plist"
codesign --force --deep --sign - "$APP"

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp -R "$APP" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Pipeline ${VERSION}" -srcfolder "$STAGE" -fs HFS+ -format UDZO -ov "$DMG" >/dev/null

(cd bin && shasum -a 256 "$(basename "$DMG")" > SHA256SUMS)
echo "package: $DMG"
cat bin/SHA256SUMS
