<script setup lang="ts">
defineProps<{ image?: string }>()
</script>

<template>
  <div class="slidev-layout epfl-full-image">
    <div class="full-image-bg" :style="{ backgroundImage: `url(${image ?? '/epfl/campus.jpeg'})` }" />
    <div class="full-image-overlay" />
    <img src="/epfl/logo.svg" class="epfl-chrome-logo" alt="EPFL" style="filter: brightness(0) invert(1); opacity: 0.95;" />
    <div class="full-image-content">
      <slot />
    </div>
  </div>
</template>

<style scoped>
.epfl-full-image { padding: 0 !important; color: white; }
.full-image-bg {
  position: absolute;
  inset: 0;
  background-size: cover;
  background-position: center;
}
.full-image-overlay {
  position: absolute;
  inset: 0;
  background: linear-gradient(180deg, rgba(0,0,0,0.0) 40%, rgba(0,0,0,0.55) 100%);
}
.full-image-content {
  position: absolute;
  left: 60px;
  right: 60px;
  bottom: 60px;
  color: white;
}
.full-image-content :deep(h1) {
  color: white;
  font-size: 2.0rem;
  font-weight: 500;
  margin: 0 0 0.3em 0;
}
</style>
