let zoneEntities = {};
let zoneCenters = {};
let zoneDataSource = null;
let predictionLabels = null;
let cesiumViewer = null;
const GEOJSON_PATH = "../Dataset/zones.geojson";
const API_BASE_URL = "http://localhost:8000"; // Re-adjusting to 8000 which is where our backend runs
const CESIUM_ACCESS_TOKEN =
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI5NDM0MjhmMi1hYjU3LTQ4ODQtYmMwOS03MjhhMDUyM2JmODQiLCJpZCI6Mjc2NjgzLCJpYXQiOjE3Mzk4MDA2NDJ9.n4Yhbtdq6trE4vfIjnG5NYmbqM7AdB64Tie_NSBbGj4"; // *** IMPORTANT: Set your Cesium Access Token ***

/**
 * Finds the zone ID whose pre-calculated center is closest to the given lat/lon.
 * @param {number} latitude Latitude of the target point.
 * @param {number} longitude Longitude of the target point.
 * @returns {number|null} The ID of the closest zone, or null if centers aren't loaded.
 */
function findClosestZoneID(latitude, longitude) {
    if (Object.keys(zoneCenters).length === 0) {
        console.error(
            "Zone centers not calculated yet. Cannot find closest zone."
        );
        return null;
    }

    const targetCartesian = Cesium.Cartesian3.fromDegrees(longitude, latitude);
    let closestZoneId = null;
    let minDistanceSq = Infinity;
    for (const zoneIdStr in zoneCenters) {
        const zoneId = parseInt(zoneIdStr, 10);
        const centerCartesian = zoneCenters[zoneId];
        const distanceSq = Cesium.Cartesian3.distanceSquared(
            targetCartesian,
            centerCartesian
        );

        if (distanceSq < minDistanceSq) {
            minDistanceSq = distanceSq;
            closestZoneId = zoneId;
        }
    }
    console.log(
        `Closest zone found: ${closestZoneId} (Distance sq: ${minDistanceSq})`
    );
    return closestZoneId;
}

/**
 * Loads the taxi zone GeoJSON, adds it to the map, and calculates zone centers.
 * @param {Cesium.Viewer} viewer The Cesium Viewer instance.
 */
async function loadZones(viewer) {
    try {
        console.log(`Loading zones from: ${GEOJSON_PATH}`);
        zoneDataSource = await Cesium.GeoJsonDataSource.load(GEOJSON_PATH, {
            stroke: Cesium.Color.BLACK.withAlpha(1.0),
            fill: Cesium.Color.WHITE.withAlpha(0.1),
            strokeWidth: 1,
            clampToGround: false,
        });

        await viewer.dataSources.add(zoneDataSource);
        predictionLabels = new Cesium.LabelCollection();
        viewer.scene.primitives.add(predictionLabels);

        const entities = zoneDataSource.entities.values;
        console.log(`Processing ${entities.length} entities from GeoJSON...`);

        entities.forEach((entity) => {
            let zoneId =
                entity.properties["LocationID"]?.getValue() ||
                entity.properties["location_id"]?.getValue() ||
                entity.properties["OBJECTID"]?.getValue();

            if (zoneId !== undefined && zoneId !== null) {
                try {
                    zoneId = parseInt(zoneId, 10);
                    if (isNaN(zoneId)) {
                        throw new Error("Parsed zoneId is NaN");
                    }

                    zoneEntities[zoneId] = entity;
                    if (entity.polygon && entity.polygon.hierarchy) {
                        const hierarchy = entity.polygon.hierarchy.getValue(
                            Cesium.JulianDate.now()
                        );
                        entity.polygon.height = 100.0;
                        if (
                            hierarchy &&
                            hierarchy.positions &&
                            hierarchy.positions.length > 0
                        ) {
                            const polyPositions = hierarchy.positions;
                            const boundingSphere =
                                Cesium.BoundingSphere.fromPoints(polyPositions);
                            zoneCenters[zoneId] = boundingSphere.center;
                        } else {
                            console.warn(
                                `Could not get polygon positions for zone ${zoneId} to calculate center.`
                            );
                        }
                    } else {
                        console.warn(
                            `Zone ${zoneId} does not have polygon geometry needed for center calculation.`
                        );
                    }
                    entity.description = `Zone ID: ${zoneId}<br>Select date/time and click 'Predict Demand'.`;
                } catch (parseError) {
                    console.warn(
                        "Could not parse zone ID for an entity:",
                        entity.properties?.getValue(Cesium.JulianDate.now()),
                        "Error:",
                        parseError
                    );
                }
            } else {
                console.warn(
                    "Could not find a valid zone ID property for an entity. Properties:",
                    entity.properties?.getValue(Cesium.JulianDate.now())
                );
            }
        });

        console.log(
            `✅ Zones loaded. ${
                Object.keys(zoneEntities).length
            } zones mapped with entities.`
        );
        console.log(
            `✅ Calculated centers for ${
                Object.keys(zoneCenters).length
            } zones.`
        );
        viewer.camera.flyTo({
            destination: Cesium.Cartesian3.fromDegrees(-74.0, 40.73, 20000.0), 
            duration: 2.0
        });
    } catch (error) {
        console.error("❌ Error loading or processing GeoJSON zones:", error);
        alert(
            `Failed to load taxi zones from ${GEOJSON_PATH}. Please check the path and file format.`
        );
    }
}
function clearPredictionLabels() {
    if (predictionLabels) {
        predictionLabels.removeAll();
    }
    for (const zoneId in zoneEntities) {
        const entity = zoneEntities[zoneId];
        if (entity && entity.polygon) {
            entity.polygon.material = Cesium.Color.WHITE.withAlpha(0.1);
            entity.polygon.extrudedHeight = 0.0;
            const originalId =
                entity.properties["LocationID"]?.getValue() ||
                entity.properties["location_id"]?.getValue() ||
                "Unknown";
            entity.description = `Zone ID: ${originalId}<br>Select date/time and click 'Predict Demand'.`;
        }
    }
    console.log("🧹 Cleared previous prediction styles.");
}

function displayPredictions(predictions, viewer, selectedDate, selectedTime) {
    if (Object.keys(zoneEntities).length === 0) {
        console.error("❌ Cannot display predictions: Zone entities not loaded.");
        return;
    }

    let totalPredictedDemand = 0;
    let displayedCount = 0;
    let errorCount = 0;
    clearPredictionLabels();
    
    // Original scaling
    const maxExpectedDemand = 150;
    
    for (const zoneIdStr in predictions) {
        try {
            const zoneId = parseInt(zoneIdStr, 10);
            const demand = predictions[zoneIdStr];

            if (isNaN(zoneId)) continue;
            if (demand < 0) {
                errorCount++;
                continue;
            }

            const entity = zoneEntities[zoneId];
            if (entity) {
                totalPredictedDemand += demand;
                displayedCount++;
                
                // Old Version logic
                const intensity = Math.min(1.0, demand / maxExpectedDemand);
                const alpha = 0.2 + intensity * 0.7; // Base alpha 0.2, max alpha 0.9
                
                if (alpha == 0.2) entity.polygon.material = Cesium.Color.WHITE;
                else if (alpha > 0.2 && alpha <= 0.5)
                    entity.polygon.material = Cesium.Color.GREEN;
                else if (alpha > 0.5 && alpha <= 0.8)
                    entity.polygon.material = Cesium.Color.YELLOW;
                else
                    entity.polygon.material = Cesium.Color.RED.withAlpha(alpha);
                
                entity.polygon.extrudedHeight = undefined; // Flat colors
                
                entity.description = `
                    <div style="font-family: 'Nunito Sans', sans-serif;">
                        <h3 style="margin:0 0 5px 0;">Zone ID: ${zoneId}</h3>
                        <p style="margin:0; font-size:16px;"><strong>${demand}</strong> rides predicted</p>
                        <hr style="border:1px solid #444; margin:8px 0;"/>
                        <p style="margin:0; font-size:12px; color:#aaa;">For ${selectedDate} ${selectedTime}</p>
                    </div>
                `;
                
                // Keep the text labels as requested by "old version"
                if(predictionLabels && zoneCenters[zoneId]){
                    predictionLabels.add({
                        position: zoneCenters[zoneId],
                        text: String(demand),
                        font: "bold 16pt Nunito Sans",
                        fillColor: Cesium.Color.BLACK,
                        outlineColor: Cesium.Color.BLACK,
                        outlineWidth: 3,
                        style: Cesium.LabelStyle.FILL_AND_OUTLINE,
                        verticalOrigin: Cesium.VerticalOrigin.CENTER,
                        horizontalOrigin: Cesium.HorizontalOrigin.CENTER,
                        pixelOffset: new Cesium.Cartesian2(0, 0),
                        eyeOffset: new Cesium.Cartesian3(0.0, 0.0, -10.0),
                        translucencyByDistance: new Cesium.NearFarScalar(1.5e3, 1.0, 5.0e4, 0.0),
                        scaleByDistance: new Cesium.NearFarScalar(1.5e3, 1.2, 5.0e4, 0.3),
                        disableDepthTestDistance: Number.POSITIVE_INFINITY,
                    });
                }
            }
        } catch (e) {
            console.error(`Error processing prediction for zone ${zoneIdStr}:`, e);
        }
    }

    const statusText = `Prediction for ${selectedDate} ${selectedTime}. <br> Displayed: ${displayedCount}.<br> Total Predicted: ${totalPredictedDemand}.`;
    $("#predictionResult").html(statusText);
    console.log(`✅ ${statusText}`);
    
    // Jump down over map
    viewer.camera.flyTo({
        destination: Cesium.Cartesian3.fromDegrees(-73.985, 40.735, 12000.0),
        orientation: {
            heading: Cesium.Math.toRadians(0.0),
            pitch: Cesium.Math.toRadians(-90.0),
            roll: 0.0
        },
        duration: 2.0,
        easingFunction: Cesium.EasingFunction.QUADRATIC_IN_OUT
    });
}

async function fetchDemandPrediction() {
    console.log("fetchDemandPrediction called");
    const selectedDate = $("#datepicker").val();
    const selectedTime = $("#timepicker").val();

    if (!selectedDate || !selectedTime) {
        $("#predictionResult").text("Please select date and time.");
        return;
    }
    const apiUrl = `${API_BASE_URL}/predict?date=${selectedDate}&time=${selectedTime}`;
    $("#predictionResult").text("Predicting demand...");
    
    let predictButton = document.getElementById("predictButton");
    predictButton.innerText = "⏳ Predicting...";
    predictButton.disabled = true;
    predictButton.style.opacity = "0.7";

    console.log(`Fetching predictions from: ${apiUrl}`);
    clearPredictionLabels();

    try {
        const response = await fetch(apiUrl);
        if (!response.ok) {
            let errorMsg = `API Error (${response.status}): ${response.statusText}`;
            throw new Error(errorMsg);
        }

        const predictions = await response.json();
        console.log("✅ Predictions received");
        displayPredictions(predictions, cesiumViewer, selectedDate, selectedTime);
    } catch (error) {
        console.error("❌ Error fetching or processing predictions:", error);
        $("#predictionResult").text(`Error: ${error.message}`);
        clearPredictionLabels();
    } finally {
        predictButton.innerText = "Predict Demand";
        predictButton.disabled = false;
        predictButton.style.opacity = "1";
    }
}
$(document).ready(function () {
    console.log("Document ready. Initializing...");
    // Intro logic will hide once data is loaded instead of randomly waiting
    console.log("Setting up main UI and Cesium...");
            $("#datepicker").datepicker({
                dateFormat: "yy-mm-dd",
                changeMonth: true,
                changeYear: true,
            });
            try {
                const now = new Date();
                const tomorrow = new Date(now);
                tomorrow.setDate(now.getDate() + 1);
                const year = tomorrow.getFullYear();
                const month = String(tomorrow.getMonth() + 1).padStart(2, "0");
                const day = String(tomorrow.getDate()).padStart(2, "0");
                $("#datepicker").val(`${year}-${month}-${day}`);
                $("#timepicker").val("12:00");
            } catch (e) {
                console.error("Error setting default date/time:", e);
                $("#datepicker").attr("placeholder", "YYYY-MM-DD");
                $("#timepicker").attr("placeholder", "HH:MM");
            }
            if (
                !CESIUM_ACCESS_TOKEN ||
                CESIUM_ACCESS_TOKEN === "YOUR_CESIUM_ACCESS_TOKEN"
            ) {
                alert(
                    "ERROR: Cesium Access Token is not set in script.js. Map functionality will be limited."
                );
                console.error("Cesium Access Token missing!");
                $("#predictionResult").text("Error: Cesium Token Missing.");
                return;
            }
            Cesium.Ion.defaultAccessToken = CESIUM_ACCESS_TOKEN;

            try {
                document.getElementById("parent").style.display = "flex"; // Ensure container has size
                cesiumViewer = new Cesium.Viewer("cesiumContainer", {
                    sceneMode: Cesium.SceneMode.SCENE2D,
                    geocoder: false,
                    animation: false,
                    timeline: false,
                    baseLayerPicker: true,
                    sceneModePicker: true,
                    selectionIndicator: false,
                    navigationHelpButton: false
                });
                // cesiumViewer.scene.globe.enableLighting = true; // Disabled for 2D mode
                cesiumViewer.scene.postProcessStages.fxaa.enabled = true;
            } catch (cesiumError) {
                console.error(
                    "FATAL: Failed to initialize Cesium Viewer:",
                    cesiumError
                );
                alert(
                    "Failed to initialize the map viewer. Check console for details."
                );
                $("#predictionResult").text("Error: Map Failed to Load.");
                return;
            }
            loadZones(cesiumViewer).then(() => {
                console.log(
                    "Zone loading process finished (or encountered errors)."
                );
                cesiumViewer.camera.flyTo({
                    destination: Cesium.Cartesian3.fromDegrees(
                        -73.97,
                        40.75,
                        18000
                    ),
                    orientation: {
                        heading: Cesium.Math.toRadians(0.0),
                        pitch: Cesium.Math.toRadians(-90.0),
                    },
                    duration: 1.0,
                });
            });
            const toolbar = document.querySelector("div.cesium-viewer-toolbar");
            if (toolbar) {
                const button = document.createElement("button");
                button.innerText = "Add 3D Buildings";
                button.classList.add("cesium-button");
                button.style.margin = "5px";
                let tileset = null;
                button.addEventListener("click", async function () {
                    if (cesiumViewer.scene.mode !== Cesium.SceneMode.SCENE3D) {
                        alert("3D Buildings can only be viewed in 3D Mode. Please switch to 3D mode using the scene picker top right.");
                        return;
                    }
                    if (tileset) {
                        cesiumViewer.scene.primitives.remove(tileset);
                        tileset = null;
                        button.innerText = "Add 3D Buildings";
                        console.log("3D Buildings Removed.");
                    } else {
                        try {
                            button.innerText = "Loading...";
                            tileset =
                                await Cesium.Cesium3DTileset.fromIonAssetId(
                                    2275207
                                );
                            cesiumViewer.scene.primitives.add(tileset);
                            button.innerText = "Remove 3D Buildings";
                            console.log("✅ 3D Buildings Loaded!");
                        } catch (error) {
                            console.error(
                                "❌ Error loading 3D Buildings tileset:",
                                error
                            );
                            alert("Failed to load 3D buildings.");
                            button.innerText = "Add 3D Buildings";
                            tileset = null;
                        }
                    }
                });
                toolbar.appendChild(button);
            } else {
                console.warn(
                    "Cesium toolbar not found, cannot add 2D/3D toggle button."
                );
            }
            const geocoderContainer =
                document.getElementById("geocoderContainer");
            if (geocoderContainer) {
                const geocoder = new Cesium.Geocoder({
                    container: geocoderContainer,
                    scene: cesiumViewer.scene,
                    viewModel: {
                        scene: cesiumViewer.scene,
                    },
                    destinationFound: function (viewModel, destination) {
                        console.log("✅ Geocode completed!");
                        let centerLon, centerLat;

                        try {
                            let targetCartographic;
                            if (destination instanceof Cesium.Cartesian3) {
                                targetCartographic =
                                    Cesium.Cartographic.fromCartesian(
                                        destination
                                    );
                            } else if (
                                destination instanceof Cesium.Rectangle
                            ) {
                                targetCartographic =
                                    Cesium.Rectangle.center(destination);
                            } else {
                                console.error(
                                    "Geocoder returned unknown destination type:",
                                    destination
                                );
                                return;
                            }

                            centerLat = Cesium.Math.toDegrees(
                                targetCartographic.latitude
                            );
                            centerLon = Cesium.Math.toDegrees(
                                targetCartographic.longitude
                            );
                            let locationID = findClosestZoneID(
                                centerLat,
                                centerLon
                            );

                            cesiumViewer.camera.flyTo({
                                destination: destination,
                                orientation: {
                                    heading: 0.0,
                                    pitch: Cesium.Math.toRadians(-90.0),
                                    roll: 0.0,
                                },
                                duration: 1.5,
                            });
                            const locationPickerInput =
                                document.getElementById("locationpicker");
                            if (locationID !== null) {
                                console.log(
                                    `📍 Mapped to PULocationID: ${locationID} (Closest Zone Center)`
                                );
                                if (locationPickerInput)
                                    locationPickerInput.value = locationID;
                            } else {
                                console.log(
                                    "Could not map geocoded point to a known taxi zone center."
                                );
                                if (locationPickerInput)
                                    locationPickerInput.value = "";
                            }
                        } catch (geoError) {
                            console.error(
                                "Error processing geocoder result:",
                                geoError
                            );
                        }
                    },
                });
                console.log("✅ Manual Geocoder added!");
            } else {
                console.warn("Geocoder container element not found.");
            }
            if ($("#predictionResult").text.length == 0) {
                $("#predictionResult").hide().empty();
            }
            let predictButton = document.getElementById("predictButton");
            if (predictButton) {
                predictButton.addEventListener("click", fetchDemandPrediction);
                console.log("✅ Predict button listener attached.");
            } else {
                console.error("❌ Predict Button not found in the DOM.");
                $("#predictionResult").text("Error: Predict button missing.");
            }
            let removeButton = document.getElementById("removePredictButton");
            if (removeButton) {
                removeButton.addEventListener("click", function () {
                    console.log("🧹 Remove Predictions button clicked!");
                    clearPredictionLabels();
                    $("#predictionResult").text("Predictions cleared.");
                });
                console.log("✅ Remove Predictions button listener attached.");
            } else {
                console.error("❌ Remove Predict Button not found in the DOM.");
            }

            // --- Wait for loadZones to finish, then dismiss intro screen ---
            const checkReady = setInterval(() => {
                if (zoneDataSource !== null) { // we know it's loaded
                    clearInterval(checkReady);
                    $("#intro-container").css("opacity", "0");
                    setTimeout(function () {
                        $("#intro-container").css("display", "none");
                    }, 500);
                }
            }, 300);

});
