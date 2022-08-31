// @ts-check

((d) => {
  /**
   * Constants
   */
  const ID_SLIDER = "slider";
  const ID_ERROR = "error";
  //
  const ID_BUTTON_PREV = "slider-action-prev";
  const ID_BUTTON_NEXT = "slider-action-next";
  //
  const ID_IMAGE_FIRST = "slider-image-first";
  const ID_IMAGE_CURRENT = "slider-image-current";
  const ID_IMAGE_FINAL = "slider-image-final";
  //
  const ID_ITERATIONS_CURRENT = "slider-iterations-current";
  const ID_ITERATIONS_FINAL = "slider-iterations-final";

  /**
   * @type {{image: string}[]}
   */
  let imageList = [];

  /**
   * @type {number}
   */
  let currentIndex = 0;

  /**
   * Initialize the Image Slider
   *
   * @param {string} jsonPayload
   */
  function initialize(jsonPayload) {
    const isOK = parseJSONPayload(jsonPayload);

    if (isOK) {
      handleSuccessState();
    } else {
      handleErrorState();
    }
  }

  /**
   * Parse the JSON payload
   *
   * @param {string} jsonPayload
   *
   * @return {boolean}
   */
  function parseJSONPayload(jsonPayload) {
    try {
      const parsedPayload = jsonPayload;

      if (Array.isArray(parsedPayload)) {
        imageList = parsedPayload;

        return true;
      }

      return false;
    } catch (err) {
      return false;
    }
  }

  /**
   * Render the application success state
   */
  function handleSuccessState() {
    /**
     * Update First Image
     */
    const imageFirst = imageList[0] ?? {};
    updateImageAttributes(ID_IMAGE_FIRST, {
      backgroundImage: imageFirst?.image ?? null,
    });

    /**
     * Update Current Image
     */
    const imageCurrent = imageList[currentIndex] ?? {};
    updateImageAttributes(ID_IMAGE_CURRENT, {
      backgroundImage: imageCurrent?.image ?? null,
    });

    /**
     * Update Final Image
     */
    const imageFinal = imageList[imageList.length - 1] ?? {};
    updateImageAttributes(ID_IMAGE_FINAL, {
      backgroundImage: imageFinal?.image ?? null,
    });

    /**
     * Update the iteration values
     */
    updateIterationValues();

    /**
     * Initialize a `click` event in the Prev Button
     */
    const $actionPrev = d.getElementById(ID_BUTTON_PREV);

    if ($actionPrev) $actionPrev.addEventListener("click", prevImageAction);

    /**
     * Initialize a `click` event in the Next Button
     */
    const $actionNext = d.getElementById(ID_BUTTON_NEXT);

    if ($actionNext) $actionNext.addEventListener("click", nextImageAction);

    /**
     * Display the Slider
     */
    const $slider = d.getElementById(ID_SLIDER);

    if ($slider) $slider.style.display = "flex";
  }

  /**
   * Render the application error state
   */
  function handleErrorState() {
    const $error = d.getElementById(ID_ERROR);

    if ($error) $error.style.display = "flex";
  }

  /**
   * Click the `Prev` image button
   */
  function prevImageAction() {
    const canGoPrev = currentIndex > 0;

    if (canGoPrev) {
      currentIndex--;

      const backgroundImage = imageList[currentIndex]?.image ?? null;

      updateImageAttributes(ID_IMAGE_CURRENT, { backgroundImage });
      updateButtonAttributes(ID_BUTTON_PREV, { disabled: false });
      updateIterationValues();

      const shouldDisable = currentIndex === 0;

      if (shouldDisable) {
        updateButtonAttributes(ID_BUTTON_PREV, { disabled: true });
      } else {
        updateButtonAttributes(ID_BUTTON_NEXT, { disabled: false });
      }
    }
  }

  /**
   * Click the `Next` image button
   */
  function nextImageAction() {
    const imageLen = imageList.length;
    const canGoNext = currentIndex < imageLen - 1;

    if (canGoNext) {
      currentIndex++;

      const backgroundImage = imageList[currentIndex]?.image ?? null;

      updateImageAttributes(ID_IMAGE_CURRENT, { backgroundImage });
      updateIterationValues();

      const shouldDisable = currentIndex === imageLen - 1;

      if (shouldDisable) {
        updateButtonAttributes(ID_BUTTON_NEXT, { disabled: true });
      } else {
        updateButtonAttributes(ID_BUTTON_PREV, { disabled: false });
      }
    }
  }

  /**
   * Update the Image attributes
   *
   * @param {ID_IMAGE_FIRST | ID_IMAGE_CURRENT | ID_IMAGE_FINAL} imageID
   * @param {{ backgroundImage: string | null}} options
   */
  function updateImageAttributes(imageID, options) {
    const { backgroundImage } = options ?? {};

    if (imageID) {
      const $img = d.getElementById(imageID);

      if ($img && backgroundImage) {
        $img.style.backgroundImage = `url("${backgroundImage}")`;
      }
    }
  }

  /**
   * Update the Prev/Next Button attributes
   *
   * @param {ID_BUTTON_PREV | ID_BUTTON_NEXT} buttonID
   * @param {{disabled: boolean}} options
   */
  function updateButtonAttributes(buttonID, options) {
    const { disabled } = options ?? {};

    if (buttonID) {
      const $button = d.getElementById(buttonID);

      // @ts-ignore
      if ($button) $button.disabled = disabled;
    }
  }

  /**
   * Update the iteration values
   */
  function updateIterationValues() {
    const $iterationCurrent = d.getElementById(ID_ITERATIONS_CURRENT);
    const $iterationFinal = d.getElementById(ID_ITERATIONS_FINAL);

    if ($iterationCurrent) {
      $iterationCurrent.innerText = `${currentIndex + 1}`;
    }

    if ($iterationFinal) {
      const len = imageList?.length ?? 0;

      $iterationFinal.innerText = `${len}`;
    }
  }

  /**
   * Trigger the `INITALIZE_IS_READY` event when the Document is ready.
   */
  d.addEventListener("DOMContentLoaded", function isReady() {
    const $body = d.querySelector("body");

    if ($body) {
      const e = new CustomEvent("INITALIZE_IS_READY", {
        detail: { initialize },
      });

      $body.dispatchEvent(e);
    }
  });
})(document);
