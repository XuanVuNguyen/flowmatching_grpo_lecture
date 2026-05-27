<script setup lang="ts">
const props = defineProps<{
  preparedBy?: string[]
}>()
</script>

<template>
  <div class="slidev-layout epfl-cover">
    <img src="/epfl/logo.svg" class="cover-logo" alt="EPFL" />

    <div class="cover-content">
      <div class="accent-bar" />
      <slot />

      <div class="cover-meta">
        <div v-if="preparedBy && preparedBy.length">
          <div class="label">Prepared by</div>
          <ul class="value prepared-list">
            <li v-for="name in preparedBy" :key="name">{{ name }}</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.prepared-list {
  list-style: none;
  padding: 0;
  margin: 0;
}
.prepared-list li {
  line-height: 1.45;
  margin: 0;
  padding: 0;
}
.prepared-list li::marker { content: none; }
</style>
